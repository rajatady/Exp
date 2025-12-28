
Continuous Thought Machines
tl;dr
Neurons in brains use timing and synchronization in the way that they compute. This property seems essential for the flexibility and adaptability of biological intelligence. Modern AI systems discard this fundamental property in favor of efficiency and simplicity. We found a way of bridging the gap between the existing powerful implementations and scalability of modern AI, and the biological plausibility paradigm where neuron timing matters. The results have been surprising and encouraging.
Luke Darlow      
Sakana AI
 
Ciaran Regan      
Sakana AI, University of Tsukuba
 
Sebastian Risi      
Sakana AI, IT University of Copenhagen
 
Jeffrey Seely      
Sakana AI
 
Llion Jones      
Sakana AI
 
MAY
2025
 
PDF
Github
Interactive demonstration
❌ Error loading image: maze_14.png. undefined.

This is the Continuous Thought Machine (CTM): a new type of neural network that uses the synchronization of neural activity over time as its representation for taking actions in a world.

This maze solving demo runs a real CTM in your browser: it tries to find a path (up to 150 steps) from the red pixel to the green pixel.

Hit the teleport button to move the start location to the last step the CTM predicts, and have fun teleporting to solve longer paths!

You can also hit the run button to watch it solve again. You can move the start or end positions by clicking on different maze locations. Toggle the move button to switch between moving start or end positions.

As it pays attention to the maze (shown below the maze and overlayed brightly) it unfolds Neural Dynamics: the complex, time-dependent patterns of neural activity. The way that neurons synchronize is how the CTM interacts with the world. Watch how it builds a route as it thinks. Have fun trying to solve these mazes quickly with the CTM!

Note that this is a much smaller model than what we trained for the full results, and it might not always perform as consistently well as what those results show.

Introduction
Neural networks (NNs) were originally inspired by biological brains, yet they remain significantly distinct from their biological counterparts. Brains demonstrate complex neural dynamics that evolve over time, but modern NNs intentionally abstract away such temporal dynamics in order to facilitate large-scale deep learning. For instance, the activation functions of standard NNs can be seen as an intentional abstraction of a neuron’s firing rate, replacing the temporal dynamics of biological processes with a single, static value. Such simplifications, though enabling significant advancements in large-scale machine learning [1, 2, 3], have resulted in a departure from the fundamental principles that govern biological neural computation.

Over hundreds of millions of years, evolution has endowed biological brains with rich neural dynamics, including spike-timing-dependent plasticity (STDP) [4] and neuronal oscillations. Emulating these mechanisms, particularly the temporal coding inherent in spike timing and synchrony, presents a significant challenge. Consequently, modern neural networks do not rely on temporal dynamics to perform compute, but rather prioritize simplicity and computational efficiency. This abstraction, while boosting performance on specific tasks, contributes to a recognized gap between the flexible, general nature of human cognition and current AI capabilities, suggesting fundamental components, potentially related to temporal processing, are missing from our current models [5, 6, 7].

Why do this research?
Indeed, the notably high performance of modern AI across many fields suggests the emulation of neural dynamics is unwarranted. However, the gap between the highly flexible and general nature of human cognition and the current state of modern AI suggests missing components in our current models.

For these reasons, we argue that time should be a central component of artificial intelligence in order for it to eventually achieve levels of competency that rival or surpass human brains [8, 9]. Therefore, in this work, we address the strong limitation imposed by overlooking neural activity as a central aspect of intelligence. We introduce the Continuous Thought Machine (CTM), a novel neural network architecture designed to explicitly incorporate neural timing as a foundational element. Our contributions are as follows:

We introduce a decoupled internal dimension, a novel approach to modeling the temporal evolution of neural activity. We view this dimension as that over which thought can unfold in an artificial neural system, hence the choice of nomenclature.
We provide a mid-level abstraction for neurons, which we call neuron-level models (NLMs), where every neuron has its own internal weights that process a history of incoming signals (i.e., pre-activations) to activate (as opposed to a static ReLU, for example).
We use neural synchronization directly as the latent representation with which the CTM observes (e.g., through an attention query) and predicts (e.g., via a projection to logits). This biologically-inspired design choice puts forward neural activity as the crucial element for any manifestation of intelligence the CTM might demonstrate.
Reasoning models and recurrence
The frontier of artificial intelligence faces a critical juncture: moving beyond simple input-output mappings towards genuine reasoning capabilities. While scaling existing models has yielded remarkable advancements, the associated computational cost and data demands are unsustainable and raise questions about the long-term viability of this approach. For sequential data, longstanding recurrent architectures [10, 11, 12] have largely been superseded by transformer-based approaches [13]. Nevertheless, recurrence is re-emerging as a natural avenue for extending model complexity. Recurrence is promising because it enables iterative processing and the accumulation of information over time. Modern text generation models (sometimes referred to as ‘reasoning models’) use intermediate generations as a form of recurrence that enables additional compute during test-time. Recently, other works have demonstrated the benefits of the recurrent application of latent layers [14, 15, 16]. While such methods bring us closer to the recurrent structure of biological brains, a fundamental gap nevertheless remains. We posit that recurrence, while essential, is merely one piece of the puzzle. The temporal dynamics unlocked by recurrence — the precise timing and interplay of neural activity — are equally crucial. The CTM differs from existing approaches in three ways: (1) the decoupled internal dimension enables sequential thought on any conceivable data modality; (2) private neuron-level models enables the consideration of precise neural timing; and (3) neural synchronization used directly as a representation for solving tasks.

Method

Fig 1. The Continuous Thought Machine: a single step in its internal recurrent process.
The CTM unfolds neural activity internally as it thinks about data. At each step (one of which demonstrated above) a truncated history of ‘pre activations’ are collected and used for the Neuron Level Models (NLMs). The history of ‘post activations’ produced by all NLMs over time are kept and used to compute neuron-to-neuron synchronization over time. This result is a Synchronization Representation: a new, parameter-efficient, and evidently powerful representation that the CTM uses to observe (via attention) and predict.
The Continuous Thought Machine (CTM) is a neural network architecture that enables a novel approach to thinking about data. It departs from conventional feed-forward models by explicitly incorporating the concept of Neural Dynamics as the central component to its functionality. The video above gives a pictorial overview of the internal workings of the CTM. We give all technical details, including additional figures and verbose explanations in our Technical Report. A GitHub repository is also available. We will provide links to relevant parts of the repository as we explain the model below.

CTM architecture
Fig 2. CTM architecture: The 1 synapse model (weights depicted as blue lines) models the cross-neuron interactions to produce pre-activations. For each neuron, a 2 history of pre-activations is kept, the most recent of which are used by the 3 neuron-level models (weights depicted as red lines) to produce 4 post-activations. A 5 history of post-activations is also kept and used to 6 compute a synchronization matrix. Neuron pairs are 7 selected from the synchronization matrix, yielding the 8 latent representations with which the CTM 9 produces outputs and modulates data through cross-attention. Modulated data (e.g., attention outputs) are 10 concatenated with post-activations for the next internal tick.
Variable	Description
Post-activations at internal tick 
, after neuron-level models have been used.
Recurrent (synapse) model weights; U-NET-like architecture that connects neurons at a given internal tick, 
.
Pre-activations at internal tick 
.
History of most recent pre-activations, designed as a FIFO list so that they are always length 
; inputs to neuron-level models.
Weights of a single neuron-level model, 
 of 
; MLP architecture, unique weights per neuron.
History of all post-activations up to this internal tick, variable length; used as input for synchronization dot products.
Synchronization matrix at internal tick 
. In practice we use far fewer neurons than 
 for separate 
 and 
 synchronization representations.
, 
Linear weight matrices that project from 
 and 
 to attention queries and predictions, respectively.
Cross attention output.
The CTM consists of three main ideas:

The use of internal recurrence, enabling a dimension over which a concept analogous to thought can occur. The entire process visualised in the video above is a single tick; the interactive maze demo at the top of the page uses 75 ticks. This recurrence is completely decoupled from any data dimensions.
Neuron-level models, that compute post-activations by applying private (i.e., on a per-neuron basis) MLP models to a history of incoming pre-activations.
Synchronization as a representation, where the neural activity over time is tracked and used to compute how pairs of neurons synchronize with one another over time. This measure of synchronization is the representation with which the CTM takes action and makes predictions. Listing 3 in the Technical Report shows the logic for this, and Appendix K details how we use a recursive computation for efficiency.
But what about data?
While data is undoubtedly crucial for any modeling, the CTM is designed around the idea of internal recurrence and synchronization, where the role of data is somewhat secondary to the internal process itself.

Input data is attended to and ingested at each internal tick based on the current sychronisation, and similarly for predictions.

Fig 3. Neural Dynamics when thinking about ImageNet: Each subplot is the activity of a single neuron over time. It is the synchronization between these that forms the representation used by the CTM.
Internal ticks: the ‘thought’ dimension
We start by introducing the continuous internal dimension: 
t∈{1,…,T}. Unlike conventional sequential models — such as RNNs or Transformers — that process inputs step-by-step according to the sequence inherent in the data (e.g., words in a sentence or frames in a video), the CTM operates along a self-generated timeline of internal thought steps. This internal unfolding allows the model to iteratively build and refine its representations, even when processing static or non-sequential data such as images or mazes. To conform with existing nomenclature used in related works [17, 18, 19, 20], we refer to these thought steps as ‘internal ticks’ from here on.

A dimension over which thought can unfold.
The CTM’s internal dimension is that over which the dynamics of neural activity can unfold. We believe that such dynamics are likely a cornerstone of intelligent thought.

Recurrent weights: synapses
A recurrent multi-layer perceptron (MLP structured in a U-NET fashion [21]) acts as a synapse model for the CTM. At any internal tick 
t, the synapse model produces what we consider pre-activations:

a
​t
​​ =f
​θ
​syn
​​ 
​​ (concat(z
​t
​​ ,o
​t
​​ ))∈ R
​D
​​ ,

where 
o
​t
​​  is from input data. The 
M most recent pre-activations are then collected into a pre-activation ‘history’:

 
A
​t
​​ =[
​a
​t−M+1
​​ 
​​ 
​a
​t−M+2
​​ 
​​ 
​⋯
​​ 
​a
​t
​​ 
​​ ]∈ R
​D×M
​​ .

Neuron-level models
M effectively defines the length of the history of pre-activations that each neuron level model works with. Each neuron, 
{1,…,D}, is then given its own privately parameterized MLP that produces what we consider post-activations:

z
​d
​t+1
​​ =g
​θ
​d
​​ 
​​ (A
​d
​t
​​ ),

where 
θ
​d
​​  are the unique parameters for neuron 
d, and 
z
​d
​t+1
​​  is a single unit in the vector that contains all post-activations. 
A
​d
​t
​​  is a 
M-dimensional vector (time series). The full set of neuron post-activations are then concatenated with attention output and fed recurrently into 
f
​θ
​syn
​​ 
​​  to produce pre-activations for next step, 
t+1, in the unfolding thought process.

Synchronization as a representation: modulating data
How should the CTM interact with the outside world? Specifically, how should the CTM consume inputs and produce outputs? We introduced a timing dimension over which something akin to thought can unfold. We also want the CTM’s relationship with data (its interaction, so to speak) to depend not on a snapshot of the state of neurons (at some 
t), but rather on the ongoing temporal dynamics of neuron activities. By way of solution, we turn again to natural brains for inspiration and find the concept of neural synchronization [22] both fitting and powerful. For synchronization we start by collecting the post-activations into a post-activation ‘history’:

 
Z
​t
​​ =[
​z
​1
​​ 
​​ 
​z
​2
​​ 
​​ 
​⋯
​​ 
​z
​t
​​ 
​​ ]∈R
​D×t
​​ .

The length of 
Z
​t
​​  is equal to the current internal tick, meaning that this dimension is not fixed and can be arbitrarily large. We define neural synchronization as the matrix yielded by the inner dot product between post-activation histories:

S
​t
​​ =Z
​t
​​ ⋅(Z
​t
​​ )
​⊺
​​ ∈ R
​D×D
​​ .

Since this matrix scales in 
O(D
​2
​​ ) it makes practical sense to subsample 
(i,j) row-column pairs, which capture the synchronization between neurons 
i and 
j. To do so we randomly select 
D
​out
​​  and 
D
​action
​​  
(i,j) pairs from 
S, thus collecting two synchronization representations, 
S
​out
​t
​​ ∈ R
​D
​out
​​ 
​​  and 
S
​action
​t
​​ ∈ R
​D
​action
​​ 
​​ . 
S
​out
​t
​​  can then be projected to an output space as:

y
​t
​​ =W
​out
​​ ⋅S
​out
​t
​​ .

Synchronization enables a very large representation.
As the model width, D, grows, the synchronization representation grows with 
 
, offering opportunities for improved expressiveness without the need for more parameters in order to project a latent space to this size.

Modulating input data
S
​action
​t
​​  can be used to take actions in the world (e.g., via attention as is in our setup):

q
​t
​​ =W
​in
​​ ⋅S
​action
​t
​​ 

where 
W
​out
​​  and 
W
​in
​​  are learned weight matrices that project synchronization into vectors for observation (e.g., attention queries, 
q
​t
​​ ) or outputs (e.g., logits, 
y
​t
​​ ). Even though there are 
(D×(D+1))/2 unique pairings in 
S
​t
​​ , 
D
​out
​​  and 
D
​action
​​  can be orders of magnitude smaller than this. That said, the full synchronization matrix is a large representation that has high future potential.

In most of our experiments we used standard cross attention [13]:

o
​t
​​ =Attention(Q=q
​t
​​ ,KV=FeatureExtractor(data))

where a ‘FeatureExtractor’ model, e.g., a ResNet [23], is first used to build useful local features for the keys and values. 
o
​t
​​  is concatenated with 
z
​t+1
​​  for the next cycle of recurrence.

Loss function: optimizing across internal ticks
The CTM produces outputs at each internal tick, 
t. A key question arises: how do we optimize the model across this internal temporal dimension? Let 
y
​t
​​ ∈R
​C
​​  be the prediction vector (e.g., probabilities of classes) at internal tick 
t, where 
C is the number of classes. Let 
y
​true
​​  be the ground truth target. We can compute a loss at each internal tick using a standard loss function, such as cross-entropy:

L
​t
​​ =CrossEntropy(y
​t
​​ ,y
​true
​​ ),

and a corresponding certainty measure, 
C
​t
​​ . We compute certainty simply as 1 - normalised entropy. We compute 
L
​t
​​  and 
C
​t
​​  for all 
t∈{1,…,T}, yielding losses and certainties per internal tick, 
L∈R
​T
​​  and 
C∈R
​T
​​ .

A natural question arises: how should we reduce 
L into a scalar loss for learning? Our loss function is designed to optimize CTM performance across the internal thought dimension. Instead of relying on a single step (e.g., the last step), which can incentivize the model to only output at that specific step, we dynamically aggregate information from two internal ticks: the point of minimum loss and the point of maximum certainty:

the point of minimum loss: 
t
​1
​​ =argmin(L); and
the point of maximum certainty: 
t
​2
​​ =argmax(C).
This approach is advantageous because it means that the CTM can perform meaningful computations across multiple internal ticks, naturally facilitates a curriculum effect, and enables the CTM to tailor computation based on problem difficulty. The final loss is computed as:

 
L=
​2
​
​L
​t
​1
​​ 
​​ +L
​t
​2
​​ 
​​ 
​​ .

More information in our Technical Report.
Please take a look at our Technical Report for more information.

Specifically, it includes additional information on how we enable the CTM to learn short versus long time dependency.

Experiment: ImageNet
Demonstrations

Pause

Unmute
Remaining Time -0:01
1x
Playback Rate

Picture-in-Picture

Fullscreen
Thumbnail for video 11037Thumbnail for video 12495Thumbnail for video 13575Thumbnail for video 14308Thumbnail for video 17352Thumbnail for video 17558Thumbnail for video 22403Thumbnail for video 23597Thumbnail for video 2677Thumbnail for video 27230Thumbnail for video 30917Thumbnail for video 3415Thumbnail for video 35303Thumbnail for video 37164Thumbnail for video 4014Thumbnail for video 4225Thumbnail for video 42416Thumbnail for video 43479Thumbnail for video 49506Thumbnail for video 7343
Fig 4. Thinking about Images: Top left is the average attention weighting (of the 16 heads shown) when the CTM observes the image on the right. Class predictions are shown on the bottom left and the certainty is shown on the bottom right (green denotes a correct prediction). The small images at the bottom are buttons to load other examples, showing a diversity of certainties and correctness.
Results
Accuracy types for ImageNet
Fig 5a. Top-5 Accuracies: using different mechanisms for predictions, the CTM achieves different levels of accuracy per internal tick (thought step). At about 15 ticks it makes sense to account for certainty.
Calibration on ImageNet validation set
Fig 5b. Calibration: often considered an important measure of how well a model fits the underlying data distribution, the CTM has remarkably good calibration.
Setting a threshold of 0.5 certainty
Fig 5c. Certainty threshold=0.5: top-5 accuracy at this certainty threshold (black line, bottom right in the videos to the left).
Setting a threshold of 0.8 certainty
Fig 5d. Certainty threshold=0.9: top-5 accuracy at this certainty threshold (black line, bottom right in the videos to the left).
This is a subset of results from our ImageNet experiments (see the Technical Report for more). Crucially, the CTM enables Adaptive Compute where the internal steps, (how much thought the CTM is putting into the problem) can be cut short. These figures show what can be expected in terms of accuracy when cutting thinking short. Only marginal gains are had past a certain point, but gains nonetheless.

Fig 4. shows where the CTM looks as it reasons about the data. We show the Attention Weights for all 16 heads and demark where the model is looking for each (and on average at the top). The predictions are shown on the bottom left and certainty over time on the bottom right. Fig 6. shows a visualization of Neural Activity as the CTM thinks about a single image: note the multi-scale structure and how activity seems to ‘flow’.

Fig 6. Neural activity: visualised in 2D using a UMAP projection. Each neuron is shown as an individual dot, scaling in size with absolute magnitude, and color with value (blue for negative, red for positive). We show similar visualizations inside later demonstrations.
Discussion
We never set out to train a model that achieved some remarkable new state-of-the-art performance on ImageNet. AI researchers already expect high performance on ImageNet after over a decade of research that uses it. Instead, we wanted to show just how different and interesting the CTM’s interaction with data can be. The videos on the left/above demonstrate the thought process the CTM undertakes and the figures show its benefits.

Let’s contextualize just what’s going on here: the CTM is looking around these images, all the while building up its prediction, all by using the synchronization of neural activity directly as a representation. The neural dynamics we showed earlier are actually examples of dynamics from a CTM observing ImageNet! The paths output by the CTM in the maze demo are akin to the class predictions made here.

The missing ingredient: TIME
Biological intelligence is still superior to AI in many cases [24, 25, 5, 26]. Biological brains solve tasks very differently to conventional neural networks, which might explain why this is the case. It might be that biological intelligence pays heed to time in ways that modern AI simply does not. In this work, we aimed to develop a model that approaches problem-solving in a manner more aligned with biological brains, emphasizing the central role of the precise timing and interplay of neural dynamics. The interpretable and intuitive outcome we point at in the video demonstrations is very exciting as it suggests that the CTM is indeed leveraging time to its advantage, in order to reason about data.

The details on model hyper-parameters can be found in the Technical Report.

Experiment: Solving 2D Mazes - doing it the hard way
The why and the how
Solving mazes is a challenging task for machines [27, 20, 28], where only the current bleeding edge models perform well on fairly simple mazes. Even so, existing methods either require careful design of the data/objective (e.g., outputs are images instead of a solution), or extensive tool use (e.g., LLMs that perform well at this), indicating that the underlying intelligent reasoning required to solve a maze, step-by-step, is not evidenced by these approaches.

We trained n CTM on a new setup, requiring it to directly predict a path (truncated for simplicity) from start to finish in the form of steps: Left, Right, Up, Down, or Wait. A small version of the resultant model can be explored in the interactive demo at the top of this page. We show a demonstration of larger model here. Remarkably, the attention pattern is intuitive and follows the solution, all while using neural synchronization as a representation. It even generalizes beyond the truncated path! See the Technical Report.

Demonstration
Fig 7. Thinking about mazes: each animation segment shows 75 internal ticks of the CTM when it is provided with the input maze. We show the route as it is constructed through the internal ‘thought process’, showing only the valid route (i.e., ignoring predictions through walls; see the associated toggle on the demo). 16 attention heads’ weights are shown at the bottom and the average is overlayed on the maze to show where the CTM is focusing. We ‘teleport’ the CTM to its resultant predicted location until it lands on the target and then load a new maze.
Results
Accuracy over training run
Fig 8a. Accuracy during training: versus the best baselines we could get working. The CTM, shown in pink, gets nearly perfect validation accuracy.
Accuracy versus path length
Fig 8b. Accuracy versus path length: the baselines are certainly learning, but the CTM far outperforms them for longer paths.
Generalization
Each video below shows how well the CTM generalizes to bigger and more complex mazes, while retaining its reasoning prowess. To generate these we used a CTM trained to solve a path up to length 100 on 39 x 39 mazes, but the mazes shown here are of size 99 x 99 and the full paths are roughly 6x as long.

Discussion
Why run these experiments? We know that neural networks can be tailored to solve 2D mazes if we present the data in the “right” way. But, when presented in a fashion that requires a clear process through which the model must progress, existing methods fall short. Even current SoTA LLMs rely on tool use, which is impressive in its own right, but somewhat unsatisfying: an intelligent machine should be demonstrably intelligent, and humans don’t require tools to solve these mazes.

We set out to show that the CTM has the capacity to learn when complex reasoning is required, unlike the most comparable baseline methods. We also show how the CTM generalizes to larger and more complex mazes, indicating that its internal reasoning is not merely memorization, but rather a more natural and correct way to solve the underlying maze problem. Importantly, we made no specific structural changes to the model compared to the CTM we trained for ImageNet; the only meaningful structural change was to output the solution as a 2D class space, applying cross entropy for each step.

A World Model
We chose our setup carefully: (1) we used no positional embedding for attention; and (2) we required that the models predict the routes directly as a string of classes (e.g., go left, left, right, up, etc.). By forgoing positional embedding the CTM must build an internal world model in order to query the data and navigate the maze. The fact that it does so in such a convincing fashion is remarkable.

Where to go from here?
We have some strong evidence that the CTM is capable of solving challenging problems, and it does so in intuitive and interesting ways. The fact that it can solve mazes by building an internal world model “on the fly” without any positional embedding opens up avenues for future research. For instance, we would like to see how the CTM finds its way around more complex environments (e.g., games or videos) without any explicit positional encodings.

Experiment: Parity
Sequential data, non-sequentially

The parity of a binary sequence, given by the sign of the product of its elements, can reasonably be predicted by an RNN when the data is fed sequentially - the model need only maintain an internal state, flipping a ‘switch’ whenever a negative number if encountered. When the entire sequence is provided at once, however, the task is significantly more challenging[29].

We trained CTMs to solve a variant of this parity task: the model is input with a 64-length binary vector, and must predict the cumulative parity at each of the 64 positions.

Demonstration
Fig 9. Determining the cumulative parity of a sequence: shown are the movements of the attention weights from each of the 8 heads. Overlayed on the input sequences is the trajectory of the attention weight argmax. The larger sequences depict the models predictions and targets.
Results
Accuracy during training
Fig 10a. Accuracy during training: versus the LSTMs, averaged over 3 training runs. The best, 75-iteration model, achieves perfect accuracy in some runs.
Accuracy versus path length
Fig 10b. Accuracy versus thinking time: more internal ticks leads to higher accuracy.
We compare the accuracy of CTMs trained with different numbers of internal ticks to parameter matched LSTMs. We found that CTMs with over 75 internal ticks could reliably solve this task, with some runs achieving 100% accuracy. The LSTMs, on the other hand, struggled to learn with over 10 internal ticks, suggesting that LSTMs are not well suited to unfolding an internal thought dimension.

The left/above demonstration shows the solving process of the CTM: the movement of the attention weights, as well as their argmax overlayed on the inputs, the models predictions, the target and the neuron activations. Notice how the attention moves backwards through the data and determines the solution after observing the entire input. Some attention heads display interpretable behavior, such as the first attention head which attends to only negative parity positions (
).

Learning sequential algorithms
We visualise the learned algorithms by plotting the accuracy (top) and attention weights (bottom) over the 75 internal ticks for each position in the 64-length sequence, at different points during training. One model (left) attends to the data in reverse order before predicting the cumulative parity at once; the other attends forward, predicting parity incrementally. Both achieve perfect accuracy.

The ability of the CTM to search through the data in reverse order, suggests that the CTM is carrying out some form of planning, building up its understanding of the data before making a final decision — the CTM is capable of forming and following a strategy.

Fig 11a. 75-Internal Tick CTM 1: learns to attend to the data in reverse order, predicting the parity at the end of the reasoning process.
Fig 11b. 75-Internal Tick CTM 2: learns to attend from beginning to end, and with it, increasing its certainty in each prediction.
Experiment: Q&A MNIST
Memory via Synchronization

To assess the CTM’s ability to memorise and recall information, we design a Question and Answering (Q&A) MNIST task. In this task, the model first observes a sequence of MNIST digits, followed by a series of interleaved index and operator embeddings that specify which digits should be recalled and which modular operation should be applied. Once all digits and index/operator embeddings have been presented, a zero-tensor flag signals the model to produce its final answer. An example is shown below.

Accuracy during training
Fig 12. Q&A MNIST example: a typical sequence observed by the model.
In our experiments, the memory length of the CTMs is such that the MNIST digits will always lie outside of the activation history window used by the neuron-level models. In this way, the CTM must organize its activations such that it can recall digits are later timesteps.

Demonstration
Fig 13.Observing digits and answering questions: the model is shown MNIST digits followed by operator and index embeddings which specifies the modular operation at the top. Shown also is the attention weights for the digits and the models predictions.
Results
Accuracy during training
Fig 14. Accuracy during training: for both CTMs and LSTMs trained with 1 internal tick per input and 10 internal ticks per input.
Our results show that, while the LSTM outperforms the CTM when only a single internal tick is used to process each input, the LSTM becomes more unstable when more internal ticks are used. The CTM, on the other hand, exhibits stronger performance with increasing internal ticks, achieving over 95% accuracy in the most challenging in-distribution task.

Furthermore, we highlight the ability of the CTM to recall digit values observed many timesteps in the past, arising purely from the organization and synchronization of neurons. This strong performance suggests that processing timing information through the synchronization of neuron activations may be a powerful mechanism for memorization and recall.

Generalization
We examine the generalization capabilities of the CTM by measuring the accuracy of the model when input with more digits or index-operator embeddings than observed during training, depicted below, with the training regime marked in red. We find that both the CTM and the LSTM baseline can generalize to an increased number of operations. Empirically, we find that this generalization arises from the model’s approach to solving the task: each time a new index embedding is presented, the model computes and stores the result of the specified operation, regardless of whether the answer flag has been given. This enables it to continue processing a stream of index and operator embeddings without needing to wait for a final signal.

Accuracy during training
Fig 15a. CTM, 1 internal tick
Accuracy during training
Fig 15b. LSTM, 1 internal tick
Accuracy during training
Fig 15c. CTM, 10 internal ticks
Accuracy during training
Fig 15d. LSTM, 10 internal ticks
Fig 15. Generalization: accuracy of the CTM and LSTM for different numbers of input digits and operations. The red line indicates the training regime. For the CTM, performance scales with the number of internal ticks, while the converse is true for the LSTM.
Additional experiments
CTM versus humans
In this section we test the CTM using CIFAR-10, comparing it to human performance, a feed-forward baseline, and an LSTM baseline. The purpose of this experiment was to contextualize the performance of the CTM alongside a standard feed-forward baseline, an LSTM baseline that also uses internal ticks for reasoning (potentially), and humans. We used a restricted backbone to highlight the differences between models (details in the Technical Report).

We used two datasets of human labels for CIFAR-10; we call these CIFAR-10D [30] owing to its calibration of difficulty levels, and CIFAR-10H [31] originally used to quantify human uncertainty. CIFAR-10D can be found at here and CIFAR-10H can be found here.

Accuracy curves during training
Fig16a. Accuracy curves during training: using parameter-matched models, the CTM generalizes best. One of the seeds had lower accuracy initially but it recovered and outperformed, interestingly, outperformed all others.
Calibration plots
Fig 16b. Calibration plots: for all models and humans. We show calibration at each step of thought for the CTM, taking the average probability up to that step for computing these. The CTM even has superior calibration than humans.
CIFAR-10D difficulty plots
Fig 16c. CIFAR-10D difficulty plots: showing how the CTM performs best at predicting difficult classes, perhaps benefiting from additional “time to think”.
LSTM pseudo 'reaction times'
Fig 16d. LSTM pseudo "reaction times": computed as (1 - the average certainty) over internal ticks, measured against real human reaction times from CIFAR-10H.
CTM pseudo 'reaction times'
Fig 16e. CTM pseudo "reaction times": while not any ‘better’ than the LSTM, this shows an interesting pattern where the CTM reacts more ‘quickly’ to challenging data.
For the human calibration we used the probabilities provided in CIFAR-10H, which were computed using guesses from multiple humans using the available human datasets. We computed calibration (Fig 16b.) as we did for ImageNet: we compute the predictive probability as the average probability for the chosen class over all internal ticks (for both CTM and LSTM). The CTM demonstrates the best calibration, even when compared to humans.

Neural dynamics of the CTM
Neural dynamics of the LSTM
Fig 17. CTM (left) and LSTM (right) neural dynamics: over internal (50) ticks. We show dynamics from other data points in the background to show how diverse these can befor the CTM. The dot products between pairs of vectors like these (not necessarily exactly these ones) is Synchronization and that is the representation used to predict the classes by the CTM.
Fig 17. shows the neural activities for the CTM and the LSTM baseline. The CTM yields rich, diverse, and complex dynamics with multiple interesting features, including periodic behavior (there is no periodic driving function). The distinct difference between the CTM and LSTM neural activities is evidence that the two novel elements of the CTM (neuron-level models and synchronization as a representation) enable neural dynamics as a fundamental computational mechanic.

CIFAR-100, ablation studies
Fig 18. shows what happens when we vary the number of neurons (i.e., the model width) while keeping all else constant, including the training time. As with other models, a wider network could evidently benefit from a longer training time or different training hyper-parameters, hence the reduction in accuracy in Fig 18a. For Fig 18b. and Fig 18c. we set out to understand how unique the Neuron-level models tend to be, and that was related to the model width, as measured by the cosine similarity between the dynamics of different neurons. Fig 18b. shows that with a wider model (i.e., more neurons), we see more diversity instead of less. One might expect that with more neurons there is less ‘space’ for diversity, but we observed the opposite.

Accuracy versus model width
Fig 18a. Accuracy versus model width: when trained on CIFAR-100. Each model had equal training, indicating that the wider models could benefit from more training.
Neuron similarity across data
Fig 18b. Neuron similarity across data: averaged over all neurons, showing how a wider model yields more diverse neurons instead of more overlap (which might be expected).
Neuron similarity across neurons
Fig 18c. Neuron similarity across neurons: averaged over data, showing a slightly reduced similarity for wider models.
Fig 19. shows the relationship between predictions and the number of internal ticks used by the CTM. We trained several CTMs (again keeping all other variables constant). In Fig 19b. we plot the distributions of the data over which steps the CTM is most certain (i.e., 
t
​2
​​  in the loss function). What this shows is that the CTM uses a wide range of steps to become most certain about the data it observes. For each setup (25, 50 and 100 internal ticks), there are two concentrated areas in the distributions, indicating that the CTM is following separate internal processes depending on the data.

Accuracy versus internal ticks
Fig 19a. Accuracy versus internal ticks: evidencing that more internal ticks might benefit from longer learning.
Histogram of most certain indices
Fig 19b.Histogram of most certain indices: for models trained using 25, 50, and 100 internal ticks. In each case there is a double ‘hump’ in the distributions of certainties, meaning that the CTM might be following two different internal processes depending on the data.
Sorting real numbers
For these experiments we trained a CTM to sort 30 real numbers from 
N(0,I
​30
​​ ). The purpose of this experiment was twofold: (1) to understand if and when the CTM applies more or less compute in a controlled environment; and (2) see if we can train the CTM to output a sequence in sequential order using the CTC loss. This CTM could sort a length 30 list of real numbers approximately 80% of the time.

Mean wait times per sequence index
Fig 20a. Mean wait times per sequence index: measured as internal ticks, showing an intersting emergent behavior where the CTM first waits (i.e., does internal compute) before outputting consistently before waiting near the end again.
Wait times versus gap to previous item
Fig 20b. Wait times versus gap to previous item: showing the relationship between how much compute the CTM applies compared to the gap between sorted items.
Generalizing beyond training distribution
Fig 20c. Generalizing beyond training distribution: showing sorting performance for different Gaussian distributions (it was trained using a Normal distribution).
Sorting demonstration'
Fig 20d. Sorting demonstration: showing the delta from mean of wait times for each item (plotted in sorted order, color denoting original order using a rainbow colormap). The CTM tends to require more compute when there is a larger gap between points.
Reinforcement Learning
We have shown that the CTM can process non-sequential data via an continuous thought dimension. Here, we extend the CTM to tasks involving interation with an external environment, training CTMs with proximal policy optimization[32] to solve a navigation task and partially observable variants of CartPole and Acrobot[33, 34]. In this setting, the CTM receives an observation, process it using a fixed number of internal thought steps, and outputs the next action. The history of activations is continuous across environment steps, such that activations from past environment steps can affect the present decision making process.

Fig 21a. CTM solving the MiniGrid Four Rooms task: evidencing that the CTM can use a leverage a continuous history of activations to interact with the world.
CTM Training Curves of MiniGrid Four Rooms
Fig 21b.Training curves: for this navigation task (episode length during training). Although the LSTM learns slightly faster, both solve the task and converge to the same average episode length.
Although our results show that the CTM achieves a comparable performance to the LSTM baseline, the central goal of this section is provide evidence that the CTM can learn in a continuous environment.

Conclusion
The Continuous Thought Machine (CTM) represents a novel step towards bridging computational efficiency with biological plausibility in artificial intelligence. By moving beyond traditional pointwise activation functions to private neuron-level models, the CTM cultivates far richer neuron dynamics. Crucially, it leverages neural synchronization as a powerful and fundamentally new type of representation - distinct from the activation vectors prevalent since the early days of neural networks. This direct use of neuron dynamics as a first-class representational citizen allows the CTM to exhibit behaviors qualitatively different from contemporary models.

Our research demonstrates the tangible benefits of this approach. The CTM can dynamically build representations over time for tasks like image classification, form rich internal maps to attend to specific input data without positional embeddings, and naturally exhibit adaptive computation. Furthermore, it learns to synchronize neural dynamics to store and retrieve memories beyond its immediate activation history. This internal processing also lends itself to greater interpretability, as seen in its methodical solving of mazes and parity tasks.

Remarkably, the core CTM architecture remained largely consistent across a diverse range of challenging tasks, requiring only input/output module adjustments. This versatility and trainability were particularly evident in complex scenarios like maze navigation. The CTM succeeded with minimal tuning, where a traditional model like the LSTMs still struggled even after significant tuning efforts.

This work underscores a vital, yet often underexplored, synergy between neuroscience and machine learning. While modern AI is ostensibly brain-inspired, the two fields often operate in surprising isolation. The CTM serves as a testament to the power of drawing inspiration from biological principles. By starting with such inspiration and iteratively following the emergent, interesting behaviors, we developed a model with unexpected capabilities, such as its surprisingly strong calibration in classification tasks, a feature that was not explicitly designed for.

It is crucial to note that our approach advocates for borrowing concepts from biology rather than insisting on strict, literal plausibility; real neurons may not access their activation history as modeled in the CTM, yet emergent phenomena like traveling waves still manifest. This nuanced balance between practicality and biological inspiration opens a landscape of new research directions, which may hold the key to unlocking capabilities currently missing in AI, potentially leading to systems that exhibit more human-like intelligence and address its current limitations.

When we initially asked, “why do this research?”, we hoped the journey of the CTM would provide compelling answers. By embracing light biological inspiration and pursuing the novel behaviors observed, we have arrived at a model with emergent capabilities that exceeded our initial designs. We are committed to continuing this exploration, borrowing further concepts to discover what new and exciting behaviors will emerge, pushing the boundaries of what AI can achieve.

Acknowledgements
Citation
For attribution in academic contexts, please cite this work as

Luke Darlow, Ciaran Regan, Sebastian Risi, Jeffrey Seely, and Llion Jones. (2025). 
Continuous Thought Machines
. Sakana AI Technical Report.
BibTeX citation

@techreport{darlow2025ctm,
  author    = {Luke Darlow and Ciaran Regan and Sebastian Risi and Jeffrey Seely and Llion Jones},
  title     = {{Continuous Thought Machines}},
  institution = {Sakana AI},
  year      = {2025},
  month     = {April},
  note      = {Technical Report}
}
Open Source Code
We release our code for this project here.

Appendix
Please view the PDF version of the paper for the appendix, which contains additional details and experiments.

References
Deep learning
LeCun, Y., Bengio, Y. and Hinton, G., 2015. nature, Vol 521(7553), pp. 436—444. Nature Publishing Group UK London.
Deep learning
Goodfellow, I., Bengio, Y., Courville, A. and Bengio, Y., 2016. , Vol 1(2). MIT press Cambridge.
Emergent abilities of large language models
Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D. and others, ., 2022. arXiv preprint arXiv:2206.07682.
Spike timing—dependent plasticity: a Hebbian learning rule
Caporale, N. and Dan, Y., 2008. Annu. Rev. Neurosci., Vol 31(1), pp. 25—46. Annual Reviews.
Building machines that learn and think like people
Lake, B.M., Ullman, T.D., Tenenbaum, J.B. and Gershman, S.J., 2017. Behavioral and brain sciences, Vol 40, pp. e253. Cambridge University Press.
Deep learning: A critical appraisal
Marcus, G., 2018. arXiv preprint arXiv:1801.00631.
On the measure of intelligence
Chollet, F., 2019. arXiv preprint arXiv:1911.01547.
Time is of the essence: neural codes, synchronies, oscillations, architectures
Cariani, P. and Baker, J.M., 2022. Frontiers in Computational Neuroscience, Vol 16, pp. 898829. Frontiers Media SA.
On the relevance of time in neural computation and learning
Maass, W., 2001. Theoretical Computer Science, Vol 261(1), pp. 157—178. Elsevier.
Long short-term memory
Hochreiter, S. and Schmidhuber, J., 1997. Neural computation, Vol 9(8), pp. 1735—1780. MIT press.
Gate-variants of gated recurrent unit (GRU) neural networks
Dey, R. and Salem, F.M., 2017. 2017 IEEE 60th international midwest symposium on circuits and systems (MWSCAS), pp. 1597—1600.
Recurrent neural networks: design and applications
Medsker, L. and Jain, L.C., 1999. CRC press.
Attention is all you need
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L. and Polosukhin, I., 2017. Advances in neural information processing systems, Vol 30.
Perceiver: General perception with iterative attention
Jaegle, A., Gimeno, F., Brock, A., Vinyals, O., Zisserman, A. and Carreira, J., 2021. International conference on machine learning, pp. 4651—4664.
Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach
Geiping, J., McLeish, S., Jain, N., Kirchenbauer, J., Singh, S., Bartoldson, B.R., Kailkhura, B., Bhatele, A. and Goldstein, T., 2025. arXiv preprint arXiv:2502.05171.
Looped transformers are better at learning learning algorithms
Yang, L., Lee, K., Nowak, R. and Papailiopoulos, D., 2023. arXiv preprint arXiv:2311.12424.
Meta learning backpropagation and improving it
Kirsch, L. and Schmidhuber, J., 2021. Advances in Neural Information Processing Systems, Vol 34, pp. 14122—14134.
Structurally Flexible Neural Networks: Evolving the Building Blocks for General Agents
Pedersen, J., Plantec, E., Nisioti, E., Montero, M. and Risi, S., 2024. Proceedings of the Genetic and Evolutionary Computation Conference, pp. 1119—1127.
Introducing symmetries to black box meta reinforcement learning
Kirsch, L., Flennerhag, S., Van Hasselt, H., Friesen, A., Oh, J. and Chen, Y., 2022. Proceedings of the AAAI Conference on Artificial Intelligence, Vol 36(7), pp. 7202—7210.
Can you learn an algorithm? generalizing from easy to hard problems with recurrent networks
Schwarzschild, A., Borgnia, E., Gupta, A., Huang, F., Vishkin, U., Goldblum, M. and Goldstein, T., 2021. Advances in Neural Information Processing Systems, Vol 34, pp. 6695—6706.
U-net: Convolutional networks for biomedical image segmentation
Ronneberger, O., Fischer, P. and Brox, T., 2015. Medical image computing and computer-assisted intervention—MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pp. 234—241.
Neural synchrony in cortical networks: history, concept and current status
Uhlhaas, P., Pipa, G., Lima, B., Melloni, L., Neuenschwander, S., Nikolic, D. and Singer, W., 2009. Frontiers in integrative neuroscience, Vol 3, pp. 543. Frontiers.
Deep residual learning for image recognition
He, K., Zhang, X., Ren, S. and Sun, J., 2016. Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770—778.
Arc prize 2024: Technical report
Chollet, F., Knoop, M., Kamradt, G. and Landers, B., 2024. arXiv preprint arXiv:2412.04604.
Humanity’s Last Exam  [link]
Phan, L., Gatti, A., Han, Z., Li, N., Hu, J., Zhang, H., Zhang, C.B.C., Shaaban, M. and others, ., 2025.
Brain-inspired Artificial Intelligence: A Comprehensive Review
Ren, J. and Xia, F., 2024. arXiv preprint arXiv:2408.14811.
T-SCEND: Test-time Scalable MCTS-enhanced Diffusion Model
Zhang, T., Pan, J., Feng, R. and Wu, T., 2025. arXiv preprint arXiv:2502.01989.
End-to-end algorithm synthesis with recurrent networks: Extrapolation without overthinking
Bansal, A., Schwarzschild, A., Borgnia, E., Emam, Z., Huang, F., Goldblum, M. and Goldstein, T., 2022. Advances in Neural Information Processing Systems, Vol 35, pp. 20232—20242.
Adaptive computation time for recurrent neural networks
Graves, A., 2016. arXiv preprint arXiv:1603.08983.
CIFAR10 to compare visual recognition performance between deep neural networks and humans
Ho-Phuoc, T., 2018. arXiv preprint arXiv:1811.07270.
Human uncertainty makes classification more robust
Peterson, J.C., Battleday, R.M., Griffiths, T.L. and Russakovsky, O., 2019. Proceedings of the IEEE/CVF international conference on computer vision, pp. 9617—9626.
Proximal policy optimization algorithms
Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O., 2017. arXiv preprint arXiv:1707.06347.
Minigrid & Miniworld: Modular & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks
Chevalier-Boisvert, M., Dai, B., Towers, M., Lazcano, R.d., Willems, L., Lahlou, S., Pal, S., Castro, P.S. and Terry, J., 2023. CoRR, Vol abs/2306.13831.
Gymnasium: A Standard Interface for Reinforcement Learning Environments  [link]
Towers, M., Kwiatkowski, A., Terry, J., Balis, J.U., Cola, G.D., Deleu, T., Goulão, M., Kallinteris, A., Krimmel, M., KG, A., Perez-Vicente, R., Pierré, A., Schulhoff, S., Tai, J.J., Tan, H. and Younis, O.G., 2024.