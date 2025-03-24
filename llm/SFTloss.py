# 在llama中微调代码
"""
preprocess_pretrain_dataset处理PreTraining阶段的数据
数据组成形式：
输入input： <bos> X1 X2 X3
标签labels：X1 X2 X3 </s>
典型的Decoder架构的数据训练方式；

preprocess_supervised_dataset处理SFT（监督微调）阶段的数据
数据组成形式：
输入input：<bos> prompt response
标签labels： -100 ... -100 response </s>
这里面labels的重点在于prompt部分的被-100所填充

对于prompt部分的labels被-100所填充，导致在计算loss的时候模型只计算response部分的loss，
-100的部分被忽略了。而这个机制得益于torch的CrossEntropyLoss ignore_index参数，
ignore_index参数定义为如果labels中包含了指定了需要忽略的类别号（默认是-100），
那么在计算loss的时候就不会计算该部分的loss也就对梯度的更新不起作用
"""

lm_logits = self.lm_head(hidden_states)
loss = None
if labels is not None:
    labels = labels.to(lm_logits.device)
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )


"""
Example:
PreTraining阶段的loss ----- SFT 阶段loss类似
input:        <bos> 我 是 一 个 中 国 人 </s>
target:       我 是 一 个 中 国 人 </s> <pad>
shift_logits: 我 是 一 个 中 国 人 </s>
"""