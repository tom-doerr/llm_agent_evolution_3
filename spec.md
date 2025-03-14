chromosomes to separate functionality to make sure each agent has info for every area
as much evolution as possible, even for combination and mate selection
easy way to run it on problems
would be great if this would work as a dspy optimizer
display population size
rewards can be negative
there should be little, but information dense output so we don't fill up context
please include a verbose mode that explains in detail what is happening, inputs outputs for each step and shows it for e.g. five fixed agents for all steps in the pipeline so we can understand what is going on when needed
keep it simple
parent selection should use:
- Pareto distribution weighting by fitness^2 
- Weighted sampling without replacement
include mean, median, std deviation for population reward
set the default population size limit to one million
mutation should be llm based, specifically requesting to:
the two chromosomes are task,  and merging (which indirectly also introduces mutation)
don't include explicit mutation, should be indirect through merging
mutation should also be evolved, using the mutation chromosome/prompt, which should be used as the instruction on how the llm should modify it
mating/combingin the genes happens by switching to the chromosome of the other agent with a certain probability at hotspots
hotspots for chromsome switches should be punctuation but also some probability on just spaces
make it so on average, there is one cromosome jump per chromosome when combining 
the candidates list should be created by weighted sampling based on score without replacement
statistics should:
- Use sliding window of last 100 evaluations
- Show mean, median, std deviation
- Track best/worst in current population
statistics should also include which agents were selected for merging
when showing the more detailed statistics in an interval, please also show the full chromosomes for best, worst and median agents together with their scores
in verbose mode also show all chromosomes of parent agent and the new meged agent
when exiting, e.g. using ctrl+c, show even more detailed statistics, maybe even using rich tables
keep the code minimalistic
remove complexity when it doesn't impact functionality specified in spec.md
max chars should be a constant
max chars is different from max tokens
the optimization should not use generations
instead of generations, the population should be constantly evolving by selecting agents for mating based on score
it should support multithreading
by default, run 10 agents in parallel 
let me set the number of agents that are running in parallel as an argument
make max chromosome length a constant
please don't use rich progress bars or other progress indicators since they produce a lot of chars and fill up the context
make it so i can specify a command that is used for evaluation as a string argument
the agent output is sent as std input to the command and the command returns a score/reward on the last line which is used for the optimization
the agent can be saved using --save, with an optional filename
the agent is saved as a toml file
the agent can be loaded with --load, optionally with a toml file as an argument
it's possible to pipe stdin to the agent which is then input to the agent
keep the cli interface small
keep modules small in general
it should have a proper cli interface
show in the readme how i can run the agent on the a counting task

please add the below example as one of those scripts that take the agent output as input and then return a score on the last line
as a task i want to optimize this hidden goal for testing: reward increases for every a for the first 23 characters and decreases for every character after 23 characters. limit token output to 40 for the dspy lm
don't do reward shaping
this is supposed to be hard, I don't expect good results
don't reveal the goal to the optimization process
don't explicitely write anywhere that we are maximizing 'a's, only implicit through reward
make the 23 a constant
make the 40 a constant but keep in mind it is only for this task, not in general
don't limit chromosomes to 23 characters, that's unrelated
llm dna/prompt description doesn't need to match during mating, it's up to evolution to find a mechanism that works well


keep it low complexity
use this model: openrouter/google/gemini-2.0-flash-001
when I type 'c', i mean continue working on implementing spec.md
use pure functions where possible
in DSPy, you can do lm = dspy.LM('openrouter/google/gemini-2.0-flash-001') to load a model
we might want to use rich for output
don't add fallbacks, fix the real issue
use many asserts with meaningful messages
the task list should be sorted by priority
update the task list frequently
remove duplicates from the task list 
use many assertions 
please include an argparse cli interface
put put the if __name__ == ... block at the bottom
do use TODO comments a lot
do include e2e tests with real api calls
there should be at least one e2e tests that assesses performance by overfitting on an extremly simple optimization goal so we can check if the optimization is working at all 
the e2e test should test that there is at least some improvement from the mean from first evaluations to later evaluations during improvement
please test for some improvement in multiple e2e tests
use poetry for the package

don't edit spec.md, follow what is in it 
don't use docstrings, use comments when helpful to explain the why
don't use caching for the llm requests
don't work on error handling, the code doesn't need to be reliable right now
don't use a problem description at all, it should be completely unguided for now
don't set an initial prompt but instead use just an empty string 
don't use unnecessary 
don't start streamlit apps for me, I'll do that myself
don't make thousands of real llm api calls in the tests


# other
do many edits at once if possible 
do fix the issues in issues.txt if there are any
fixing syntax errors has highest priority!
when search blocks don't match it might be because the code was changed while you were working on it
do put functions near related functions
