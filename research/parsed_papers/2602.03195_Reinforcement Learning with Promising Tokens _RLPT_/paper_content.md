# Reinforcement Learning with Promising Tokens (RLPT)
- **File:** `2602.03195v2.pdf`
- **Pages:** 14
- **Paper ID:** `2602.03195`

---

## Page 1

Reinforcement Learning with Promising Tokens for Large Language Models
A PREPRINT
Jing-Cheng Pang 1 Liang Lu 1 Xian Tang 1 Kun Jiang 1 Sijie Wu 1 Kai Zhang 1 Xubin Li 1
Abstract
Reinforcement learning (RL) has emerged as a
key paradigm for aligning and optimizing large
language models (LLMs). Standard approaches
treat the LLM as the policy and apply RL di-
rectly over the full vocabulary space. However,
this formulation includes the massive tail of con-
textually irrelevant tokens in the action space,
which could distract the policy from focusing on
decision-making among the truly reasonable to-
kens. In this work, we verify that valid reason-
ing paths could inherently concentrate within a
low-rank subspace. Based on this insight, we
introduce Reinforcement Learning with Promis-
ing Tokens (RLPT), a framework that mitigates
the action space issue by decoupling strategic
decision-making from token generation. Specifi-
cally, RLPT leverages the semantic priors of the
base model to identify a dynamic set of promising
tokens and constrains policy optimization exclu-
sively to this refined subset via masking. Theoreti-
cal analysis and empirical results demonstrate that
RLPT effectively reduces gradient variance, stabi-
lizes the training process, and improves sample ef-
ficiency. Experiment results on math, coding, and
telecom reasoning show that RLPT outperforms
standard RL baselines and integrates effectively
across various model sizes (4B and 8B) and RL
algorithms (GRPO and DAPO).
1. Introduction
Developing large language models (LLMs) capable of learn-
ing and improving autonomously is a hallmark of machine
intelligence. Reinforcement Learning (RL) has emerged as
the primary paradigm for this objective, enabling LLMs to
improve by optimizing against explicit reward signals that
measure the quality of the model’s response (Pang et al.,
2024a; Mu et al., 2024). By formulating the LLM as a
policy, RL enables the maximization of long-term rewards,
1Huawei Technologies Co., Ltd.. Correspondence to: Jing-
Cheng Pang <pangjcheng@gmail.com>.
I get stuck in heavy 
traffic and risk missing 
my flight. Now I see a 
subway station nearby. 
What should I do?
Contextual text
Cry
Hope
Abandon
Data
Leave
bag
car
taxi
ticket
apple
for
happy
at
@
and
Irrelevant
Promising
Selected
...
Figure 1. An illustration of decision making with promising tokens.
At each step, the policy selects tokens solely from a high-likelihood
subset, enabling it to focus on strategic decision-making.
clearly enhancing performance in tasks requiring complex
reasoning, such as math and coding (Shao et al., 2024).
However, directly applying standard RL algorithms (e.g.,
PPO (Schulman et al., 2017), REINFORCE (Sutton et al.,
1999)) to LLMs presents a unique challenge regarding the
high dimensionality of the action space (Wen et al., 2024).
Unlike traditional RL tasks with compact action spaces
(e.g., Atari, Robot control (Mnih et al., 2013; Todorov et al.,
2012)), the action space of an LLM corresponds to its full
vocabulary, typically exceeding 50,000 tokens. In prac-
tice, previous works have attempted to alleviate this issue
using inference strategies such as Top-k (Hopkins et al.,
2023) or Nucleus sampling (Holtzman et al., 2020), which
prune the token space while maintaining text coherence dur-
ing the rollout phase (Sheng et al., 2024). Yet, the policy
optimization step still operates over the vocabulary size,
creating a off-policy mismatch between efficient rollout and
high-dimensional policy optimization. Consequently, the
RL signal is distracted by the need to maintain basic syn-
tactic coherence rather than focusing solely on the logical
reasoning required for improvement.
To bridge this gap, we suggest that RL for LLM policies
should operate within a more focused decision space, as
illustrated in Fig. 1. We begin by investigating the dis-
tribution of successful trajectories and empirically verify
that the tokens required for correct solutions are naturally
concentrated within a small subspace of high-likelihood can-
didates. Building on this insight, we propose Reinforcement
Learning with Promising Tokens (RLPT) method, which de-
couples logical decision-making from token generation. The
1
arXiv:2602.03195v2  [cs.LG]  14 Feb 2026


---

## Page 2

Reinforcement Learning with Promising Tokens for Large Language Models
core intuition of RLPT is to enable the LLM policy to make
decisions over the subspace of the token space, and focus
more on strategic selection. To achieve this, RLPT leverages
the semantic priors of the pre-trained base model to iden-
tify a dynamic set of promising tokens, i.e., candidates with
high likelihood that ensure syntactic correctness. At each
generation step, RLPT constructs a refined action space that
restricts the RL process to optimize solely over these promis-
ing tokens via masking. By filtering out the long tail tokens,
RLPT shifts the burden of the policy from predicting the
next word to selecting the best move, allowing the agent to
prioritize logical reasoning over the syntactic maintenance.
Our contributions are summarized as follows. We introduce
the concept of learning with promising tokens, a paradigm
that restricts policy optimization to a semantically valid
subspace to improve learning efficiency. Our empirical anal-
ysis provides a strong empirical foundation for pruning the
action space. Besides, we propose RLPT, a practical frame-
work that implements this concept by dynamically masking
irrelevant tokens during both the rollout and training phases.
Furthermore, we provide a theoretical analysis demonstrat-
ing that optimizing over the constrained promising set signif-
icantly reduces the variance of the policy gradient estimator,
thereby stabilizing training. Finally, extensive experiments
on mathematical reasoning and coding tasks demonstrate
that RLPT achieves superior performance and sample effi-
ciency compared to strong RL baselines, including GRPO
(Shao et al., 2024) and DAPO (Yu et al., 2025).
2. Preliminary
RL for LLM optimization. We consider a standard lan-
guage modeling task in which an LLM serves as a policy
πθ parameterized by θ. Given a context input (prompt or
question) x, the model generates a response y composed of a
sequence of tokens y = (y1, y2, . . . , yT ), where each token
yt belongs to a fixed vocabulary space V. The probability
of generating the entire sequence is given by the chain rule:
πθ(y | x) = QT
t=1 πθ(yt | x, y<t), where y<t denotes the
partial sequence generated prior to step t.
To align the LLM with human preferences or specific logical
rules, we formulate the generation process as a Markov
Decision Process (MDP (Puterman, 1994)) defined by the
tuple ⟨S, A, T , R, γ⟩:
• State space S: The state st at time step t consists of
the prompt and the history of generated tokens, i.e.,
st = (x, y1, . . . , yt−1);
• Action space A: the space of the tokens V;
• Transition T : st+1 = st ∪{at};
• Reward function R: a score r(st, at) that reflects the
quality of the generated answer to the question, typically
given at the end of the sequence (e.g., rT = R(x, y))
based on correctness verification, with intermediate re-
wards being zero;
• Discount factor γ: We typically set γ = 1 for episodic
generation tasks.
The goal of RL training is to find an optimal policy πθ that
maximizes the expected cumulative reward over the prompt
distribution D:
J(θ) = Ex∼D,y∼πθ(·|x)[R(x, y)],
where R(x, y) represents the reward (e.g., answer correct-
ness) received at the end of the episode.
3. Analyzing the Potential for Decision Making
over Promising Tokens
Before introducing our method, we answer an important
question: Is the subspace of promising tokens sufficient for
LLMs to solve the problem? Intuitively, while the vocabulary
of an LLM is vast (|V| > 50k), the tokens required for a
logically correct step are often concentrated within a small
subset of high-probability candidates. This is supported by
the successful application of top-k sampling trick. In this
section, we conduct an empirical analysis using the Qwen
series models (Bai et al., 2023) to answer this question. We
first formally define the concept of promising tokens.
Definition 3.1 (Promising Tokens).
Given the current
state st and the policy distribution π(·|st), let rank(v | st)
denote the rank of a token v ∈V when sorted in descending
order of its probability. The set of promising tokens, denoted
as Pt, is defined as the top-K candidates:
Pt = {v ∈V | rank(v | st) ≤N} .
(1)
Consequently, |Pt| = K and Pt ⊂V.
This definition constructs a focused action space, filtering
out the long tail of contextually irrelevant tokens. To verify
the coverage of Pt, we analyze the token ranks of successful
trajectories from two distinct perspectives: the ground truth
answer and the model’s own successful answer.
Setup. We employ Qwen3-8B and Qwen3-32B as base
models. We use three diverse datasets: GSM8K (Cobbe
et al., 2021) for mathematics, HumanEval (Chen, 2021) for
coding, and AlpacaEval (Taori et al., 2023) for general in-
struction following. For each step t in a successful trajectory
y, we compute the rank of the target token yt within the
base model’s predicted distribution πbase(·|y<t). We define
the top-K coverage rate as the percentage of tokens in these
trajectories that satisfy yt ∈Pt.
3.1. Analysis of Ground-truth Solution
Observation 1: The labeled solution is effectively con-
tained within the promising token subspace.
2


---

## Page 3

Reinforcement Learning with Promising Tokens for Large Language Models
Table 1. Top-K coverage rate (%) of successful trajectory tokens
from labeled solutions. Math, Code, and General denote GSM8K,
HumanEval, and AlpacaEval dataset, respectively.
Metric
Qwen3-8B
Qwen3-32B
Math
Code
General
Math
Code
General
Top-2
91.5
92.0
83.0
92.1
92.7
83.8
Top-4
95.3
95.0
91.0
96.1
95.3
91.7
Top-8
97.4
96.5
95.4
98.1
96.7
95.9
Top-16
98.6
97.2
97.6
99.0
97.5
98.0
Top-32
99.2
97.7
98.8
99.5
98.0
99.0
We first investigate whether the ground truth answers (la-
beled solutions) fall within the promising tokens predicted
by the base model. This measures the feasibility of finding
the optimal policy within the promising token space Pt. As
shown in Tab. 1, the correct answer tokens are highly con-
centrated in the head of the distribution. For Qwen3-32B,
over 99.5% of the ground truth tokens fall within the Top-32
candidates for Math tasks. Even for the smaller 8B model,
the Top-32 coverage consistently exceeds 97% across all
domains. Besides, qualitative inspection reveals that the few
outliers (i.e., tokens outside Pt) mainly appear at the very
beginning of the answer generation or consist of infrequent
symbols, typically having negligible impact on the core
reasoning logic. This observation confirms that the optimal
moves required to solve the problem are already statistically
prioritized by the pre-trained model. Therefore, pruning
the vocabulary to Pt does not impose an upper bound on
performance, as the solution space remains intact.
3.2. Analysis of Model-generated Solution
Observation 2: The model’s intrinsic answering capabil-
ities do not rely much on the long tail of the vocabulary.
While previous experiment confirms that the labeled solu-
tion is largely covered by promising tokens, one might worry
that the model needs to explore creative low-probability to-
kens to find its own path to the solution. To address this, we
sample multiple answers from the base model and analyze
only those that reach the correct final answer. Tab. 2 reports
the rank of tokens in these model-generated successful paths.
The results show that for Math and Code tasks, 100% of
the tokens in correct solutions are located within the Top-8
candidates. This indicates that when the model successfully
reasons, it rely most on the high-likelihood tokens. The
long tail tokens contribute virtually nothing to correct rea-
soning chains. Consequently, the challenge of RL may not
be expanding the search to the full vocabulary, but to learn
to decide the correct move from the promising tokens.
4. Method
In this section, we present RLPT, a framework designed
to improve RL optimization for LLMs by constraining the
Table 2. Top-K coverage rate (%) of successful trajectory tokens
from model-generated solutions.
Metric
Qwen3-8B
Qwen3-32B
Math
Code
General
Math
Code
General
Top-2
98.9
98.1
93.2
92.1
97.8
92.4
Top-4
99.9
99.6
98.0
100
100
100
Top-8
100
100
100
100
100
100
optimization landscape to a semantically valid subspace,
as shown in the overall framework in Fig. 2. The core
mechanism of RLPT operates in a two-stage process: first,
constructing a binary mask Mt based on the semantic priors
of the behavior policy; second, applying this mask to decou-
ple the logical decision-making from the massive vocabulary
space during both the sampling and training phases.
4.1. Promising Token Construction
The foundation of RLPT is the identification of promising
tokens. At each time step t, given the current state st, we
utilize the behavior policy πold (the policy used for data
collection) to compute the probability distribution over the
vocabulary V.
We formally define the Promising Mask Mt ∈0, 1|V| as a
binary vector that filters out contextually irrelevant tokens.
Let Pt be the set of top-K tokens determined by πold(·|st).
The elements of the mask vector Mt are defined as:
Mt[v] = I(v ∈Pt) =
(
1,
if v ∈Pt
0,
otherwise ,
(2)
where I(·) is the indicator function. This mask filter for the
generation ensures that subsequent operations focus exclu-
sively on candidates that are syntactically and semantically
plausible according to the base model’s priors.
4.2. Policy Rollout and Optimization with Promising
Token Masking
RLPT integrates the promising mask into the standard RL
pipeline by modifying both the rollout (sampling) and the
policy update (training) processes.
Policy Rollout.
During the data generation phase, we aim
to explore diverse reasoning paths while maintaining co-
herence. Instead of sampling from the full distribution,
we sample from a masked distribution ˜π. Specifically, the
probability of selecting a token at is renormalized over the
promising set:
˜πold(at | st) =
πold(at | st) · Mt[at]
P
v∈V πold(v | st) · Mt[v].
(3)
This operation is computationally equivalent to performing
Top-K sampling. Crucially, the mask vector Mt alongside
3


---

## Page 4

Reinforcement Learning with Promising Tokens for Large Language Models
Large Language
Model
1 0 0 0 0
0
0
1
1 1 1
Token masking &
renormalization
𝝅‘(∙|𝒔𝒕) ∝𝝅(∙|𝒔𝒕) ∙M𝒕
Identify promising
tokens
Promising token
mask 𝑀𝑡
Policy Optimization
Policy gradient over
promising tokens
Full vocabulary distribution
𝝅(∙|𝒔𝒕)
Normalized distribution
𝝅′(∙|𝒔𝒕)
Trajectory & Reward
Sampled token 𝒂𝒕
Update model
parameters
Rollout action
selection
17, 19, 21. Do all these 
numbers add up to an 
odd number?
Prompt
Complete?
Yes
No
RLPT Process
Standard LLM Process
Figure 2. The overall framework of RLPT method.
the trajectory could be stored for the training phase.
Policy Optimization.
A key innovation of RLPT is consis-
tent masking during optimization. Standard RL maximizes
the likelihood of actions across the entire vocabulary, in-
troducing noise from the long tail of irrelevant logits. In
contrast, RLPT optimizes the policy πθ to select the best
token within the promising subspace.
During the backward pass, we apply the pre-computed mask
Mt (derived from the behavior policy) to the current policy’s
logits. The gradient updates are calculated using the masked
probability:
˜πθ(at | st) = Softmax (Logitsθ(st) + (1 −Mt) · (−∞))at .
(4)
By forcing the probability mass of masked tokens to zero,
we ensure that the policy is not penalized for fluctuations in
the logits of irrelevant tokens.
4.3. Integration with RL Algorithms
RLPT is algorithm-agnostic and can be seamlessly inte-
grated with arbitrary RL objectives. We illustrate this using
Group Relative Policy Optimization (GRPO) (Shao et al.,
2024) as a representative baseline. Given a group of trajec-
tories Traj1, . . . , TrajG sampled with the masked policy,
the objective function of RLPT-GRPO is formulated as:
JRLPT(θ) =ET raj∼˜πold
"
1
T
T
X
t=1
min
 
˜πθ(at|st)
˜πold(at|st)At,
clip
 ˜πθ(at|st)
˜πold(at|st), 1 −ϵ, 1 + ϵ

At
!#
,
(5)
where At is the advantage computed from rewards. The
key difference from standard GRPO is that the probability
ratio ˜πθ
˜πold is computed strictly within the promising subspace
defined by Mt. This ensures that the policy improvement
step is perfectly aligned with the exploration step.
4.4. Theoretical Justification
In this subsection, we provide a theoretical analysis to justify
why optimizing over a constrained promising set Pt leads to
more stable training compared to the full vocabulary space
V. We focus on the variance of the policy gradient estimator,
which is a key factor in the convergence speed and stability
of RL algorithms.
Consider the gradient of the objective function J(θ) with
respect to the logits zt ∈R|V| of the policy at step t. The
standard policy gradient estimator ˆg can be expressed as:
ˆg = ∇zt log πθ(at|st) · At,
(6)
where At is the advantage function. For the softmax pa-
rameterization π(a|s) =
eza
P
v∈V ezv , the gradient of the log-
probability with respect to the logit vector z is given by
∇z log π(a|s) = ea −π, where ea is the one-hot vector for
action a and π is the probability vector over V.
Let T = V \ Pt denote the set of “tail” tokens that are
masked out in RLPT. We analyze the variance of the gradient
estimator by decomposing the contribution of the promising
set Pt and the tail set T .
Proposition 4.1 (Variance Reduction). Assuming the ad-
vantage At is bounded, optimizing the policy over the con-
4


---

## Page 5

Reinforcement Learning with Promising Tokens for Large Language Models
strained space Pt strictly reduces the variance of the gradi-
ent estimator associated with the tail tokens T , compared
to optimization over the full vocabulary V.
We present the proof of Prop. 4.1 in Appendix B.1.
The theoretical analysis guarantees that RLPT strictly re-
duces the variance of the gradient estimator by eliminating
the stochastic fluctuations arising from the high-dimensional
tail, providing a stable optimization landscape for efficient
policy learning.
5. Related Work
In this section, we review related work from the following
three areas.
5.1. Reinforcement Learning for LLM Training
RL has became the standard method for aligning LLMs with
human intent and enhancing their reasoning capabilities
(Pang et al., 2026; DeepSeek-AI, 2024; Pang et al., 2024b).
Early works, such as RLHF (Ouyang et al., 2022), employ
the PPO algorithm (Schulman et al., 2017) to optimize poli-
cies with respect to learned reward models. DPO (Rafailov
et al., 2023) and its variants (Azar et al., 2024; Ethayarajh
et al., 2024) simplify this process by implicitly optimizing
the reward function, yet they fundamentally rely on the same
policy gradient formulation. Beyond general alignment, RL
has shown remarkable efficacy in reasoning-intensive do-
mains. Methods like GRPO (Shao et al., 2024) and RFT
(Trung et al., 2024) leverage outcome-based supervision
to improve mathematical and code reasoning. However,
a common limitation across these approaches is that they
typically formulate the policy optimization problem over
the entire vocabulary space. This high-dimensional action
space introduces gradient variance and potentially explores
a vast number of contextually irrelevant tokens, making the
training process sample-inefficient and unstable. This work
addresses this fundamental inefficiency by redefining the
optimization landscape from the full vocabulary to a refined
subspace of promising candidates.
5.2. Decoding-time Action Pruning Strategies
Managing the vast action space of LLMs is a widely stud-
ied problem during LLM inference. Deterministic strate-
gies, such as greedy decoding, often yield repetitive loops
(Pipis et al., 2025), while stochastic methods, such as Top-
k sampling (Hopkins et al., 2023) and nucleus sampling
(Holtzman et al., 2020), are widely used to truncate the tail
of the probability distribution. These methods effectively
prune implausible tokens to ensure linguistic coherence and
diversity during text generation. However, these pruning
strategies are primarily used as decoding heuristics during
the rollout phase. The subsequent policy optimization step
in standard RL algorithms (Sheng et al., 2024; Havrilla et al.,
2023) typically ignores this structure, either by calculating
gradients based on the full policy distribution or by failing
to theoretically align the training objective with the pruned
search space. This discrepancy creates a train-inference mis-
match: the model explores within a constrained subspace
but updates its parameters as if it had a global action space.
We discuss this mismatch in Appendix A. Unlike these
heuristic approaches, RLPT integrates token pruning into
the RL training loop, theoretically justifying optimization
over a dynamic subset of tokens to reduce variance.
5.3. Reinforcement Learning with Large Action Space
Handling large or continuous action spaces is a longstand-
ing challenge in RL research. Wolpertinger method (Zhong
et al., 2018) embeds actions in a continuous space to han-
dle a large discrete action space. Yet, it assumes gradual
change between actions. In domains such as RTS games or
code generation, invalid-action masking (Huang & Onta˜n´on,
2022) is often used to manually filter out illegal moves based
on rigid rules or syntax constraints. While effective, these
methods rely on static, domain-specific rules that are dif-
ficult to scale to the open-ended semantic space of natural
language. An exception is ASRE (Pang et al., 2025), which
automatically identifies sparse action to constrain its execu-
tion, but it is hard to handle the vast action space. Another
line of research explores hierarchical RL (HRL) (Nachum
et al., 2018; Pang et al., 2023), which decomposes tasks into
high-level planning and low-level execution. However, ap-
plying HRL to token-level generation often yields complex
architectures that are difficult to tune. In contrast, RLPT pro-
poses a lightweight approach that leverages the LLM’s prior
knowledge to handle the large action space.
6. Experiment
In this experiment, we conduct comprehensive experiments
to evaluate the effectiveness of the proposed RLPT method.
The main goal of our experiments is to answer the following
key research questions: (1) How does RLPT compare with
standard RL algorithms in terms of answer accuracy? (Sec.
6.2) (2) How does RLPT improve the stability and quality
of the RL training process? (Sec. 6.3) (3) Is implicit token
pruning superior to training an explicit selection module?
(Sec. 6.4) and (4) How does the size of the promising token
set affect performance? (Sec. 6.5) We begin by introducing
the experimental setting.
6.1. Experimental Setting
Datasets for evaluation. To verify the effectiveness of
the proposed RLPT method, we evaluate our approach on
diverse benchmarks covering three representative domains:
5


---

## Page 6

Reinforcement Learning with Promising Tokens for Large Language Models
Table 3. Performance of various methods on diverse datasets. The values in the table represent the accuracy (%) of the answers for
different tasks. The reported results are evaluated by averaging 4 samples for each question.
Method
Math
Telecom
Code
Average
Math-17k
AIME-24
AIME-25
Datacom
Wireless
OpenR1-Code
No Training
25.2
16.7
16.7
55.65
50.87
40.89
34.34
GRPO
34.7
20.0
19.3
50.87
52.17
40.18
36.20
GRPO+RLPT
38.3
23.3
18.0
51.30
55.65
40.92
37.91
DAPO
36.4
20.7
17.3
54.43
49.57
39.87
36.38
DAPO+RLPT
39.7
19.3
20.7
54.78
50.43
39.01
37.32
• Mathematical reasoning: We utilize Math-17k (Yu et al.,
2025), AIME-24 and AIME-25 (Hendrycks et al., 2021)
to evaluate the model’s capacity for multi-step logical
deduction. These tasks serve as a rigorous testbed for
precision in token selection, where even minor stochastic
deviations in early reasoning steps can lead to catastrophic
error propagation.
• Code: We leverage OpenR1-Code (Hugging Face, 2025)
to evaluate proficiency in algorithmic synthesis. Unlike
natural language, code generation necessitates navigating
deterministic syntax and long-range functional dependen-
cies, providing a benchmark for the model’s structural
coherence.
• Telecom: We also consider telecommunication tasks
to verify broader application of RLPT method, includ-
ing Datacom and Wireless. We construct a dataset for
each task comprising 12,000 question-answer pairs, syn-
thesized from a diverse corpus of product documenta-
tion, technical solutions, and domain knowledge bases.
The datasets encompass diverse question types, includ-
ing single-choice, multiple-choice, and open-ended QA,
covering fundamental principles, product concepts, termi-
nology understanding, and multi-step reasoning tasks.
Appendix C.1 shows examples of the input and output of
these datasets.
Baseline RL methods. In our experiments, we select two
representative RL methods as baselines: Group Relative
Policy Optimization (GRPO (Shao et al., 2024)), and De-
coupled Clip and Dynamic Sampling Policy Optimization
(DAPO (Yu et al., 2025)). GRPO eliminates the critic model
by estimating advantages through group-wise normalization
of rewards. DAPO builds upon this framework by introduc-
ing dynamic sampling, which filters out prompt groups with
identical rewards to ensure effective gradient updates, and
decoupled clipping to mitigate entropy collapse.
Implementation details. Experiments are implemented us-
ing the MindSpeed-RL framework (Feng et al., 2025). We
employ Qwen3-4B and Qwen3-8B as base models. Reward
signals are provided by rule-based verifiers for Math, ex-
ecution sandboxes for Code, and model-based evaluation
(Deepseek-V3 (DeepSeek-AI, 2024)) for Telecom. Unless
stated otherwise, we use a promising set size of K = 4 and
report results at 200 training steps. Computing resources
includes clusters with 256 KUNPENG 920 CPUs and 8 As-
cend 910B3 NPUs. Full hyperparameters are in Appendix
C.3. We refer the readers to Appendix C for more details
about the experimental setting.
6.2. Main Results
Performance on mathematical tasks. Tab. 3 presents the
comparative results on the Qwen3-8B backbone. RLPT con-
sistently outperforms baselines across all domains, with
the most significant gains observed in mathematical reason-
ing. When applied to the GRPO baseline, RLPT achieves a
substantial improvement of 3.6% on the Math-17k dataset
(rising from 34.7% to 38.3%) and a 3.3% gain on AIME-
24. While the improvements on AIME-25 are mixed, the
overall trend suggests that pruning low-likelihood tokens
effectively reduces the burden for maintaining the textual
coherence on complex math problems. This empirical ev-
idence strongly validates our core hypothesis in Sec. 1:
mathematical problem-solving involves searching through a
vast combinatorial space of reasoning steps. Standard RL
algorithms often waste sample efficiency exploring syntacti-
cally valid but logically irrelevant tokens (the “long tail”).
By constraining the policy optimization to the promising
subspace, RLPT effectively filters out this noise, allowing
the agent to allocate its exploration budget exclusively to
high-value logical decisions.
Generalization across domains. Beyond mathematics,
RLPT demonstrates robust generalization. In the Code do-
main (i.e., OpenR1-Code), it maintains a competitive edge,
balancing the need for rigid syntax with algorithmic logic.
In the Telecom domain, RLPT improves over DAPO by
0.86% on Wireless tasks. These results confirm that iden-
tifying promising tokens is a domain-agnostic advantage
that enhances sample efficiency regardless of the underlying
knowledge domain.
6


---

## Page 7

Reinforcement Learning with Promising Tokens for Large Language Models
0
50
100
150
200
Training Steps
0.2
0.4
0.6
Training Accuracy
GRPO w/ RLPT
GRPO
DAPO w/ RLPT
DAPO
Figure 3. Training curves on Math-17k dataset.
Training Efficiency. As visualized in Fig. 3, RLPT not only
achieves higher asymptotic performance but also demon-
strates superior sample efficiency. The reward curve rises
sharply in the early stages compared to GRPO. This acceler-
ation suggests that by removing the burden of maintaining
syntactic coherence (which is already handled by the pre-
trained prior), RLPT enables the policy to “cut to the chase,”
focusing immediately on optimizing reasoning trajectories.
Training with Qwen3-4B on Math-17k. To verify that the
effectiveness of RLPT is not confined to a specific model
capacity, we conduct an ablation study with the Qwen-4B
model on the Math-17k dataset. The results, presented in
Fig. 4, demonstrate that RLPT consistently yields perfor-
mance improvements on Qwen3-4B. As observed, RLPT
provides a steady gain over the standard GRPO algorithm.
Specifically, it achieves an absolute accuracy improvement
of 1.3% on the 4B model and a more pronounced 3.6% on
the 8B model. This consistent trend across different scales
suggests that our proposed reinforcement learning strategy
is robust and scales effectively with increased model capac-
ity, further facilitating the discovery of successful reasoning
trajectories.
Qwen-4B
Qwen-8B
0
10
20
30
40
Accuracy (%)
23.1
25.2
30.8
34.7
32.1
38.3
No Training
GRPO
RLPT
Figure 4. Performance of RLPT method on different sizes of
models.
0
50
100
150
200
Training Steps
0.10
0.15
0.20
0.25
Gradient Norm
RLPT
GRPO
Figure 5. Gradient norm curves of GRPO and GRPO+RLPT during
the training process.
6.3. Performance Analysis of RLPT Method
To better understand why RLPT improves training efficiency,
we conduct a deeper analysis focusing on gradient variance
and qualitative decision-making.
Gradient norm during the training process. A key the-
oretical advantage of RLPT is the reduction of variance in
policy gradient estimation. By masking out the long tail
of irrelevant tokens, the policy avoids assigning probability
mass to invalid actions, thereby reducing noise in gradient
updates. We visualize the gradient norm during the training
of GRPO and GRPO+RLPT in Fig. 5. We observe that
the gradient norm of RLPT method is more stable and con-
sistently lower in compared to the baseline. The baseline
exhibits spikes, indicating unstable updates where the policy
attempts to correct for large shifts in the probability of tail
tokens. In contrast, RLPT maintains a smoother optimiza-
tion trajectory, confirming that constraining the action space
acts as an effective regularizer.
Case study on decision making with promising token. To
verify that the promising tokens correctly capture meaning-
ful decision space, we examine specific inference steps from
the GSM8K dataset (Cobbe et al., 2021) using the trained
model. Tab. 4 illustrates representative cases. Given a con-
text requiring a reasoning step, the promising set contains
distinct logical connectors, e.g., so, since, but. .. . Different
selections could lead to diverse reasoning directions. Cru-
cially, syntactically irrelevant words (e.g., “apple”, “dog”)
are successfully filtered out. This observation confirms that
RLPT shifts the RL agent’s focus from what fits the sentence
to reasoning with the proper scenario, effectively raising
the abstraction level of the policy’s decisions.
6.4. Comparison with Explicit Selector Policy for Token
Selection
As the core idea of this work is to enable the policy to fo-
cus on decision making, a natural alternative is to train an
explicit, lightweight selector policy to filter tokens, rather
than using our implicit probability-based masking. We im-
7


---

## Page 8

Reinforcement Learning with Promising Tokens for Large Language Models
Table 4. Case study on the decision making over promising tokens.
Contextual text
Promising tokens
A robe takes 2 bolts of blue fiber
and half that much white fiber.
How many bolts in total does it
take? Answer: It takes 2/2=1
bolt of white fiber.
[Total, It, So, The,
But, A, That, In]
Every day, Wendi feeds each
of her chickens three cups of
mixed chicken feed, containing
seeds, mealworms and vegeta-
bles to help keep them healthy.
She gives the chickens their feed
in three separate meals ... 60
cups of feed per day.
[She, In, Since, To-
tal, So, They, Thus,
Each]
plement this baseline by training a multilayer perceptron
selector that takes the context embedding and candidate
token embeddings as input to output the index for token
selection. We pre-train this selector on ALPACA (Taori
et al., 2023), a general QA dataset, to learn general token
relevance.
We compare this explicit selector method with RLPT on
Math-17k, using GRPO as the base optimization algorithm.
The results are shown in Fig. 6. We observe that the Ex-
plicit Selector method stops improving at a significantly
lower score (˜25%) than RLPT (˜40%). This failure can be
attributed to the fact that the external selector is initialized
from scratch, losing the general knowledge from LLM pre-
training. In contrast, RLPT leverages the intrinsic semantic
priors of the base model itself, ensuring that the promising
set is always naturally aligned with the model’s capabilities.
0
50
100
150
200
Training Steps
0.0
0.2
0.4
0.6
Reward Score
Implicit
Explicit
Figure 6. Comparison of RLPT training with explicit or implicit
token selector.
6.5. Ablation Study
Training with different sizes of promising set. The hyper-
parameter K controls the trade-off between the expressive-
ness of the action space and the efficiency of exploration.
0
20
40
60
80
100
Training Steps
0.0
0.2
0.4
Training Accuracy
K=4
K=8
K=16
K=32
Figure 7. Ablation study on the size of promising set (K).
We evaluate K ∈4, 8, 16, 32 on Math-17k, as shown in
Fig. 7. Interestingly, we find that a compact set of K = 4
yields the best asymptotic performance. Increasing K to
32 results in a noticeable performance drop, regressing to-
wards the baseline. This supports our premise that the “true”
reasoning path typically lies within the top few candidates.
Expanding K unnecessarily reintroduces the noise of the
large vocabulary, diluting the RL signal. Therefore, a tight
constraint (K = 4 ∼8) strikes the optimal balance: it is
large enough to include the correct reasoning step (i.e., high
recall) but small enough to enforce focused decision-making
(i.e., high signal-to-noise ratio).
7. Conclusion & Future Work
In this work, we proposed RLPT, a RL framework that
decouples strategic decision-making from basic token gen-
eration. By restricting the RL policy to a dynamic set of
promising tokens identified by the base model’s seman-
tic priors, we effectively transform the high-dimensional
vocabulary space into a manageable decision space. Our
theoretical analysis and empirical evaluations on mathe-
matical reasoning and coding tasks demonstrate that RL
with promising tokens clearly reduces gradient variance and
improves sample efficiency.
Despite the improvements in training stability and perfor-
mance, there are still some limitations. Currently, RLPT re-
lies on a simple top-k ranking to define the promising token
set. While effective, fixed-size subsets may be sub-optimal.
Future work could explore more adaptive thresholding meth-
ods such as top-p sampling or min-p sampling (Nguyen
et al., 2024), which might better capture the dynamic nature
of token uncertainty across different reasoning steps. Re-
stricting the action space carries the inherent risk of exclud-
ing critical decision tokens from the candidate set. While
our empirical analysis shows that successful trajectories
are over 99% covered by the top-32 tokens, the missing
< 1% could be necessary for complex reasoning or tasks
with complicated symbols. We hope that future research
will explore these intriguing questions and contribute to
developing LLMs that could learn more effectively.
8


---

## Page 9

Reinforcement Learning with Promising Tokens for Large Language Models
References
Azar, M. G., Guo, Z. D., Piot, B., Munos, R., Rowland, M.,
Valko, M., and Calandriello, D. A general theoretical
paradigm to understand learning from human preferences.
In AISTATS, 2024.
Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan,
Y., Ge, W., Han, Y., Huang, F., et al. Qwen technical
report. arXiv preprint, abs/2309.16609, 2023.
Chen, M. Evaluating large language models trained on code.
arXiv preprint, abs/2107.03374, 2021.
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H.,
Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano,
R., et al. Training verifiers to solve math word problems.
arXiv preprint, abs/2110.14168, 2021.
DeepSeek-AI.
Deepseek-v3 technical report.
arXiv,
abs/2412.19437, 2024.
Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., and
Kiela, D. KTO: model alignment as prospect theoretic
optimization. arXiv preprint, abs/2402.01306, 2024.
Feng, L., Pan, C., Guo, X., Mei, F., Ning, B., Zhang, J., Liu,
X., Zhou, B., Shu, Z., Liu, C., Yang, G., Han, Z., Wang,
J., and Wang, B. Mindspeed RL: distributed dataflow for
scalable and efficient RL training on ascend NPU cluster.
arXiv, abs/2507.19017, 2025.
Havrilla, A., Zhuravinskyi, M., Phung, D., Tiwari, A., Tow,
J., Biderman, S., Anthony, Q., and Castricato, L. trlx: A
framework for large scale reinforcement learning from
human feedback. In EMNLP, 2023.
Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart,
S., Tang, E., Song, D., and Steinhardt, J. Measuring
mathematical problem solving with the MATH dataset.
In NeurIPS Datasets and Benchmarks, 2021.
Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y.
The curious case of neural text degeneration. In ICLR,
2020.
Hopkins, A. K., Renda, A., and Carbin, M. Can llms gen-
erate random numbers? evaluating llm sampling in con-
trolled domains. In ICML SODS Workshop, 2023.
Huang, S. and Onta˜n´on, S. A closer look at invalid action
masking in policy gradient algorithms. In FLAIRS, 2022.
Hugging Face. Open r1: A fully open reproduction of
deepseek-r1, January 2025. URL https://github.
com/huggingface/open-r1.
Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A.,
Antonoglou, I., Wierstra, D., and Riedmiller, M. A. Play-
ing atari with deep reinforcement learning. arXiv preprint,
abs/1312.5602, 2013.
Mu, T., Helyar, A., Heidecke, J., Achiam, J., Vallone, A.,
Kivlichan, I., Lin, M., Beutel, A., Schulman, J., and
Weng, L. Rule based rewards for language model safety.
In NeurIPS, 2024.
Nachum, O., Gu, S., Lee, H., and Levine, S. Data-efficient
hierarchical reinforcement learning. In NeurIPS, 2018.
Nguyen, M. N., Baker, A., Neo, C., Roush, A., Kirsch,
A., and Shwartz-Ziv, R. Turning up the heat: Min-p
sampling for creative and coherent llm outputs. arXiv
preprint, abs/2407.01082, 2024.
Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright,
C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K.,
Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L.,
Simens, M., Askell, A., Welinder, P., Christiano, P. F.,
Leike, J., and Lowe, R. Training language models to
follow instructions with human feedback. In NeurIPS,
2022.
Pang, J., Yang, S., Chen, X., Yang, X., Yu, Y., Ma, M.,
Guo, Z., Yang, H., and Huang, B. Object-oriented option
framework for robotics manipulation in clutter. In IROS,
2023.
Pang, J., Wang, P., Li, K., Chen, X., Xu, J., Zhang, Z., and
Yu, Y. Language model self-improvement by reinforce-
ment learning contemplation. In ICLR, 2024a.
Pang, J., Yang, S., Li, K., Zhang, J., Chen, X., Tang, N.,
and Yu, Y. KALM: knowledgeable agents by offline rein-
forcement learning from large language model rollouts.
In NeurIPS, 2024b.
Pang, J., Xu, T., Jiang, S., Liu, Y., and Yu, Y. Reinforce-
ment learning with sparse-executing action via sparsity
regularization. IEEE Transactions on Neural Networks
and Learning Systems, 36(9):16072–16084, 2025.
Pang, J.-C., Sun, L., Zhou, C., Tang, X., Ma, H., Jiang,
K., Wang, J., Zhang, K., Wu, S., Cai, H., et al.
Edco: Dynamic curriculum orchestration for domain-
specific large language model fine-tuning. arXiv preprint,
abs/2601.03725, 2026.
Pipis, C., Garg, S., Kontonis, V., Shrivastava, V., Kr-
ishnamurthy, A., and Papailiopoulos, D.
Wait, wait,
wait... why do reasoning models loop? arXiv preprint,
abs/2512.12895, 2025.
Puterman, M. L. Markov Decision Processes: Discrete
Stochastic Dynamic Programming. Wiley Series in Prob-
ability and Statistics. Wiley, 1994.
9


---

## Page 10

Reinforcement Learning with Promising Tokens for Large Language Models
Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D.,
Ermon, S., and Finn, C. Direct preference optimization:
Your language model is secretly a reward model.
In
NeurIPS, 2023.
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and
Klimov, O. Proximal policy optimization algorithms.
arXiv preprint, abs/1707.06347, 2017.
Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Zhang, M.,
Li, Y. K., Wu, Y., and Guo, D. Deepseekmath: Pushing
the limits of mathematical reasoning in open language
models. arXiv preprint, abs/2402.03300, 2024.
Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R.,
Peng, Y., Lin, H., and Wu, C. Hybridflow: A flexible and
efficient rlhf framework. arXiv preprint, abs/2409.19256,
2024.
Sutton, R. S., McAllester, D. A., Singh, S., and Mansour, Y.
Policy gradient methods for reinforcement learning with
function approximation. In NeurIPS, 1999.
Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X.,
Guestrin, C., Liang, P., and Hashimoto, T. B. Alpaca: A
strong, replicable instruction-following model. Stanford
Center for Research on Foundation Models, 3(6):7, 2023.
Todorov, E., Erez, T., and Tassa, Y. Mujoco: A physics
engine for model-based control. In IROS, 2012.
Trung, L. Q., Zhang, X., Jie, Z., Sun, P., Jin, X., and Li,
H. Reft: Reasoning with reinforced fine-tuning. In ACL,
2024.
Wen, M., Wan, Z., Wang, J., Zhang, W., and Wen, Y. Rein-
forcing LLM agents via policy optimization with action
decomposition. In NeurIPS, 2024.
Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai,
W., Fan, T., Liu, G., Liu, L., et al. DAPO: An open-
source llm reinforcement learning system at scale. arXiv
preprint, abs/2503.14476, 2025.
Zhong, C., Gursoy, M. C., and Velipasalar, S. A deep rein-
forcement learning-based framework for content caching.
In CISS, 2018.
10


---

## Page 11

Reinforcement Learning with Promising Tokens for Large Language Models
Appendix
A. Discussion
A.1. The Off-policy Issue for LLM Optimization with Standard RL
we discuss a theoretical inconsistency prevalent in standard RL baselines regarding the mismatch between the behavior
policy (used for rollout) and the target policy (used for optimization).Standard RL approaches for LLMs (e.g., PPO) typically
employ decoding strategies such as Top-k or Nucleus (Top-p) sampling during the data collection (rollout) phase to prevent
the generation of incoherent text from the long tail. Let πθ denote the parameterized policy over the full vocabulary V. The
actual behavior policy πb during rollout is a truncated version:
πb(at|st) = Truncate(πθ(at|st); k, p),
where the support of πb is strictly smaller than V. However, during the optimization phase, standard baselines typically
calculate policy gradients (and KL-divergence penalties) based on the original distribution πθ over the full vocabulary.
This discrepancy turns standard baselines into an unacknowledged off-policy setting with biased estimators. In contrast,
RLPT resolves this inconsistency by integrating the token masking directly into the policy definition. By optimizing
over the same masked subspace used for generation, we ensure that the behavior policy and the optimization policy are
mathematically consistent, adhering to the on-policy assumption.
B. Omitted Proof
B.1. Proof of Proposition 4.1
Proposition 4.1 (Variance Reduction) Assuming the advantage At is bounded, optimizing the policy over the constrained
space Pt strictly reduces the variance of the gradient estimator associated with the tail tokens T , compared to optimization
over the full vocabulary V.
Proof. First, we calculate the variance of the gradient component for a single token i. Since at ∼π(·|st), the term I(at = i)
is a Bernoulli variable with parameter πi = π(i|st). The variance of ˆgi is:
Var(ˆgi) = Var ((I(at = i) −πi)At) = πi(1 −πi)A2
t.
(7)
For the standard optimization over the full vocabulary V, the total variance sums over all tokens:
V(ˆgfull) =
X
i∈V
πi(1 −πi)A2
t =
X
i∈Pt
πi(1 −πi)A2
t +
X
k∈Tt
πk(1 −πk)A2
t.
(8)
In RLPT, the policy is renormalized to ˜π over Pt, and the logits for tail tokens k ∈Tt are masked (i.e., gradients are
deterministically zero). Thus, Var(ˆgmask,k) = 0 for all k ∈Tt. The total variance becomes:
V(ˆgmask) =
X
i∈Pt
˜πi(1 −˜πi)A2
t.
(9)
Since the tail probability mass is negligible (as empirically verified), we have ˜πi ≈πi for i ∈Pt. The reduction in variance
is dominated by the removal of the tail sum:
∆V = V(ˆgfull) −V(ˆgmask) ≈
X
k∈Tt
πk(1 −πk)A2
t.
(10)
Since πk > 0 for tokens in the tail (due to the Softmax properties) and A2
t ≥0, the term P
k∈Tt πk(1 −πk)A2
t represents a
strictly positive quantity. This explicitly shows that standard RL optimization suffers from noise injection caused by the
gradients of thousands of irrelevant tail tokens, which RLPT effectively eliminates.
11


---

## Page 12

Reinforcement Learning with Promising Tokens for Large Language Models
C. More Details about Experiment Settings
C.1. Examples of the Datasets in Our Experiments
Tab. C.1 presents the examples from the datasets used for training and evaluation in our experiments.
Dataset
Example Question
Target Answer
Math17k
“In triangle ABC, sin ∠A = 4
5 and ∠A < 90◦. Let D
be a point outside triangle ABC such that ∠BAD =
∠DAC and ∠BDC = 90◦. Suppose that AD = 1
and that BD
CD = 3
2. If AB + AC can be expressed in
the form a
√
b
c
. . . ”
“The answer is 34.”
AIME24
“Every morning Aya goes for a 9-km-long walk and
stops at a coffee shop. When she walks at s km/h, the
walk takes 4 hours, including t minutes in the shop.
When she walks s+2 km/h, it takes 2h 24m, including
t minutes. If she walks at s + 1
2 km/h, find the total
minutes ...”
“The answer is 204.”
AIME25
“In △ABC, D, E ∈AB and F, G ∈AC. Given
AD = 4, DE = 16, EB = 8 and AF = 13, FG =
52, GC
=
26.
Let M be the reflection of D
through F, and N be the reflection of G through
E. If Area(DEGF) = 288, find the area of heptagon
AFNBCEM.”
“The answer is 588.”
OpenR1-Code
“Polycarp has n different binary words. He wants to
reverse minimal number of words so that the final set
can be arranged in a game sequence where each word
starts with the previous word’s last character ...”
“if zo > oz:
(zo-oz)//2
...else:
(oz-zo)//2
...”
Table 5. Examples from the mathematical reasoning datasets.
C.2. Prompts Used in Our Experiments
Tab. 7 presents the prompts used in our exeriments.
C.3. Hyperparameters
The hyper-parameters for implementing RLPT and experiments are presented in Tab. 8. When implementing baseline
methods, we use the same hyper-parameters as RLPT.
12


---

## Page 13

Reinforcement Learning with Promising Tokens for Large Language Models
Domain
Example question
Target answer
Wireless-training
“What should I do if the DSP GTP-
PATH command output shows that the
GTP path is in the DEETECT state
when the alarm is generated?”
”...1.First, check whether the
peer GSN address specified
in the alarm information is
valid...;2.Next,
execute
the
PING command to check if the
link
is
normal...;3.Confirm
whether the peer GSN can
respond to ECHO messages. .. ”
Wireless-testing
“During the deployment and opera-
tion of the LMT, the LMT is forcibly
connected in HTTPS or WSS mode
to ensure secure connection. In this
mode, digital certificates are required
for authentication. In addition, ...In
such a deployment scenario, if the
MAE of a carrier is set to HTTP login
mode and the LMT is set to forcible
HTTPS connection mode, what will
happen? A. ...B. If the LMT con-
nection mode is set to Force HTTPS,
MAE proxy login fails to access the
LMT due to protocol mismatch. The
connection cannot be established even
if the OM channel is normal. C. . . . D.
...”
“The answer is X.”
Datacom-training
“When configuring the Segment
VXLAN feature, how can you enable
EVPN as the VXLAN control plane
on Transit Leaf1 and Transit Leaf2,
and configure BGP EVPN peer rela-
tionships?”
”...1.Enter
the
BGP
view
or
BGP
multi-instance
view...;2.Enter
the
BGP-
EVPN
address
family
view...;3.Configure
the
split group for BGP EVPN
peers (groups)...;4.Enable the
function to mark routes received
from BGP EVPN peers as
re-originated...”
Datacom-testing
“...The device supports creating
subinterfaces on Layer 2 Ethernet
and Layer 2 Eth-Trunk interfaces for
VLAN termination to achieve inter-
VLAN forwarding.
However, the
USG9500 series devices do not sup-
port creating subinterfaces on these
two types of interfaces.”
“The answer is error”
Table 6. Examples from the datasets used in our experiments.
13


---

## Page 14

Reinforcement Learning with Promising Tokens for Large Language Models
Table 7. Prompts for Experiments.
Task Name
Prompt
Math
Let’s think step by step. Output the final answer within
\boxed{}, e.g., \boxed{5}.
Code
Solve the following coding problem using the program-
ming language python: {problems}
Telecom
For the following multiple-choice question, there is
only one correct answer. Please analyze the question
and the options, and place the correct option ID within
\boxed{}, e.g., \boxed{A}.
Table 8. Hyper-parameters for training RLPT and baselines.
Hyper-parameters
Value
Batch size
8
Learning rate
10−6
Learning rate decay style
cosine
Train iteration
200
Sequence length
4096
RL γ
1
RL λ
0.95
Mini batch size
4
RL clip ratio
0.2
Promising set size
4
Entropy coefficient
0.0
KL coefficient
0.01
Rollout count
8
14


---
