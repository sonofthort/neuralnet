# Neural Net
Some neural net code. Examples of training a neural net to create game ai.

Right now all C++ with OpenMP support.

Only implements a feed forward net now and has some genetic algorithm examples for evolving the net.

# TODO
- Add backpropagation support.
- Add recurrent neural net implementation
- Support dynamically sized nets.
	- Make a new FeedForwardNet base class that use arrays only and CRTP.
	- Will support parallel weight update.
