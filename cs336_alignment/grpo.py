import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import Callable
from torch.optim import Optimizer

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    """
    将 prompt 和 output 分别分词后拼接，并生成与 shifted labels 对齐的 response_mask。
    """
    batch_size = len(prompt_strs)
    all_concat_ids = []
    prompt_lens = []
    full_lens = []

    # 分别对 prompt 和 output 分词
    for p_str, o_str in zip(prompt_strs, output_strs):
        p_ids = tokenizer.encode(p_str, add_special_tokens=False)
        o_ids = tokenizer.encode(o_str, add_special_tokens=False)
        concat = p_ids + o_ids
        
        all_concat_ids.append(concat)
        prompt_lens.append(len(p_ids))
        full_lens.append(len(concat))

    max_len = max(full_lens)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    input_ids_list = []
    labels_list = []
    mask_list = []

    for i in range(batch_size):
        concat = all_concat_ids[i]
        p_len = prompt_lens[i]
        f_len = full_lens[i]

        padded = concat + [pad_id] * (max_len - f_len)

        # input_ids: 切掉最后一个 token
        input_ids_list.append(padded[:-1])
        # labels: 切掉第一个 token
        labels_list.append(padded[1:])
        
        # response_mask: 对齐 shifted labels
        # labels 中的第 idx 个 token，对应原 padded 中的位置 idx + 1
        mask = [1 if (p_len <= (idx + 1) < f_len) else 0 for idx in range(max_len - 1)]
        mask_list.append(mask)

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "response_mask": torch.tensor(mask_list, dtype=torch.long),
    }

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    计算模型在每个位置预测 label 的对数概率 log p_theta(labels | input_ids)，
    可选返回每个位置的预测信息熵（entropy）。
    """
    # 1. 模型前向传播获取 logits
    logits = model(input_ids).logits 

    # 2. 计算所有词的对数概率
    log_probs_dist = F.log_softmax(logits, dim=-1) 

    # 3.  提取真实 labels 对应的 token 的 log_prob
    # labels 原形状: (batch_size, seq_len) -> unsqueeze 变为 (batch_size, seq_len, 1)
    gathered_log_probs = torch.gather(
        log_probs_dist, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)  # 还原形状 (batch_size, seq_len)

    res = {"log_probs": gathered_log_probs}

    # 4.计算每个 token 分布的信息熵 H = - sum(p * log p)
    if return_token_entropy:
        probs = torch.exp(log_probs_dist)
        # 对应位置相乘后沿词表维度求和
        entropy = -torch.sum(probs * log_probs_dist, dim=-1)  # 形状: (batch_size, seq_len)
        res["token_entropy"] = entropy

    return res    

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    计算一批采样回答的原始奖励，并统计元数据。
    """
    raw_rewards = []
    format_rewards = []
    answer_rewards = []

    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        scores = reward_fn(resp, gt)
        
        r = scores.get("reward", scores.get("answer_reward", 0.0))
        raw_rewards.append(r)
        
        format_rewards.append(scores.get("format_reward", 0.0))
        answer_rewards.append(scores.get("answer_reward", 0.0))

    raw_rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32)

    metadata = {
        "mean_reward": float(raw_rewards_tensor.mean().item()),
        "mean_format_reward": float(torch.tensor(format_rewards).mean().item()),
        "mean_answer_reward": float(torch.tensor(answer_rewards).mean().item()),
    }
    return raw_rewards_tensor, metadata

from typing import Literal

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    按题目组（Group）对原始奖励进行基线减除和优势归一化。
    """
    total_samples = raw_rewards.shape[0]
    n_groups = total_samples // group_size

    # 1. 变形
    rewards_grouped = raw_rewards.view(n_groups, group_size)

    # 2. 计算基线（行均值）
    group_mean = rewards_grouped.mean(dim=1, keepdim=True)

    if baseline == "mean":
        adv = rewards_grouped - group_mean
    elif baseline == "none":
        adv = rewards_grouped.clone()
    else:
        raise NotImplementedError(f"Unsupported baseline: {baseline}")

    # 3. 归一化除法
    if advantage_normalizer == "std":
        
        group_std = rewards_grouped.std(dim=1, keepdim=True)
        adv = adv / (group_std + advantage_eps)
    elif advantage_normalizer == "mean":
        adv = adv / (group_mean.abs() + advantage_eps)
    elif advantage_normalizer == "none":
        pass
    else:
        raise NotImplementedError(f"Unsupported advantage_normalizer: {advantage_normalizer}")

    # 4. 展平成原始一维形状 
    advantages = adv.view(total_samples)

    metadata = {
        "mean_advantage": float(advantages.mean().item()),
        "std_advantage": float(advantages.std().item()),
    }
    return advantages, metadata

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    计算每个 token 处的策略梯度损失（ on-policy, noclip, grpo, gspo）。
    """
    if raw_rewards_or_advantages.ndim == 1:
        adv = raw_rewards_or_advantages.unsqueeze(-1)
    else:
        adv = raw_rewards_or_advantages

    metadata = {}

    # 1. On-policy
    if importance_reweighting_method == "none":
        per_token_loss = -adv * policy_log_probs

    # 2. 不裁剪的 Token 级重要性采样 (w_t = exp(log_pi - log_pi0))
    elif importance_reweighting_method == "noclip":
        assert old_log_probs is not None
        ratio = torch.exp(policy_log_probs - old_log_probs)
        per_token_loss = -ratio * adv

    # 3. PPO / GRPO 标准 Token 级双向裁剪
    elif importance_reweighting_method == "grpo":
        assert old_log_probs is not None
        assert cliprange is not None
        ratio = torch.exp(policy_log_probs - old_log_probs)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * adv
        per_token_loss = -torch.min(surr1, surr2)

        # 统计裁剪比例
        is_clipped = (ratio < (1.0 - cliprange)) | (ratio > (1.0 + cliprange))
        metadata["clip_fraction"] = is_clipped.float()

    # 4. GSPO 序列级几何平均裁剪 
    elif importance_reweighting_method == "gspo":
        assert old_log_probs is not None
        assert cliprange is not None
        assert response_mask is not None

        # 计算整条回答的对数比率之和，并除以有效回答长度（几何平均）
        diff = (policy_log_probs - old_log_probs) * response_mask
        seq_lens = response_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        seq_log_ratio = diff.sum(dim=-1, keepdim=True) / seq_lens  # 形状: (batch_size, 1)
        s_seq = torch.exp(seq_log_ratio)

        # 序列级裁剪
        surr1 = s_seq * adv
        surr2 = torch.clamp(s_seq, 1.0 - cliprange, 1.0 + cliprange) * adv
        clipped_obj = torch.min(surr1, surr2)

        # 梯度形式为 s * (1/L) * sum(grad log_prob)
        per_token_loss = -clipped_obj.expand_as(policy_log_probs)

        is_clipped = (s_seq < (1.0 - cliprange)) | (s_seq > (1.0 + cliprange))
        metadata["clip_fraction"] = is_clipped.float()

    else:
        raise NotImplementedError(f"未知的方法: {importance_reweighting_method}")

    return per_token_loss, metadata

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """
    根据掩码和指定的归一化策略，将每个 token 的损失聚合成一个标量损失。
    """
    # 1. 过滤掉题目与 padding 处的损失
    masked_loss = per_token_policy_gradient_loss * mask

    if loss_normalization == "sequence":
        # 统计每条回答的有效 token 长度  clamp 防止除以 0
        seq_lens = mask.sum(dim=-1).clamp(min=1.0)
        # 每条序列先求各自的 token 平均损失
        seq_loss = masked_loss.sum(dim=-1) / seq_lens
        # 再对整个 batch 的序列求平均
        return seq_loss.mean()

    elif loss_normalization == "constant":
        # 常数归一化：所有有效 token 损失求和后除以固定的 normalization_constant
        return masked_loss.sum() / normalization_constant

    else:
        raise NotImplementedError(f"不支持的归一化方式: {loss_normalization}")

def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    
    total_rollouts = len(repeated_prompts)

    # 1. 计算原始奖励
    raw_rewards, reward_meta = compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths
    )

    # 2. 组归一化计算优势 Advantage
    advantages, adv_meta = compute_group_normalized_rewards(
        raw_rewards=raw_rewards,
        group_size=group_size,
        baseline=baseline,
        advantage_eps=advantage_eps,
        advantage_normalizer=advantage_normalizer,
    )

    # 3. 标准版：对全部样本统一分词
    tokenized = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    response_mask = tokenized["response_mask"]

    device = model.device

    # 4. 微批次切分
    microbatch_size = total_rollouts // gradient_accumulation_steps

    accumulated_loss = 0.0
    all_entropies = []
    optimizer.zero_grad()

    # 5. 循环每个微批次计算梯度
    for i in range(0, total_rollouts, microbatch_size):
        mb_input_ids = input_ids[i:i + microbatch_size].to(device)
        mb_labels = labels[i:i + microbatch_size].to(device)
        mb_mask = response_mask[i:i + microbatch_size].to(device)
        mb_adv = advantages[i:i + microbatch_size].to(device)
        mb_old_lp = old_log_probs[i:i + microbatch_size].to(device) if old_log_probs is not None else None

        # 计算当前微批次的对数概率与信息熵
        log_prob_res = get_response_log_probs(
            model, mb_input_ids, mb_labels, return_token_entropy=True
        )
        mb_log_probs = log_prob_res["log_probs"]
        
        valid_tokens = mb_mask.sum().clamp(min=1.0)
        entropy_val = ((log_prob_res["token_entropy"] * mb_mask).sum() / valid_tokens).item()
        all_entropies.append(entropy_val)

        # 计算单点策略梯度损失
        per_token_loss, _ = compute_policy_gradient_loss(
            raw_rewards_or_advantages=mb_adv,
            policy_log_probs=mb_log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=mb_old_lp,
            cliprange=cliprange,
            response_mask=mb_mask,
        )

        # 累积缩放方式：
        # microbatch_loss * (len(inputs_microbatch) / len(inputs))
        if loss_normalization == "sequence":
            mb_loss = aggregate_loss_across_microbatch(
                per_token_loss, mb_mask, loss_normalization="sequence"
            ) * (mb_input_ids.shape[0] / total_rollouts)
        else:
            mb_loss = aggregate_loss_across_microbatch(
                per_token_loss, mb_mask, loss_normalization="constant", normalization_constant=normalization_constant
            )

        mb_loss.backward()
        accumulated_loss += mb_loss.item()

    # 6. 梯度裁剪与优化器步进
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
    else:
        grad_norm = 0.0

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    metadata = {
        **reward_meta,
        **adv_meta,
        "loss": accumulated_loss,
        "grad_norm": grad_norm,
        "token_entropy": sum(all_entropies) / len(all_entropies) if all_entropies else 0.0,
    }
    return torch.tensor(accumulated_loss), metadata