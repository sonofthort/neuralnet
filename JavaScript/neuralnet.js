var neuralnet = {}

neuralnet.sigmoid = function(x) {
	return 1 / (1 + Math.exp(-x))
}

neuralnet.Net = function(sizes, initializer) {
	var depth = sizes.length - 1,
		dataSize = sizes[0],
		numWeights = 0
	
	for (var i = 0; i < depth; ++i) {
		var numInputs = sizes[i],
			numOutputs = sizes[i + 1] // num neurons
			
		numWeights += (numInputs + 1) * numOutputs // +1 for bias
		dataSize += numOutputs
	}
	
	this.sizes = sizes.slice()
	this.weights = new Float32Array(numWeights)
	this.data = new Float32Array(dataSize) // stores all the inputs/ouputs for each neuron from the last calculation
	
	if (typeof initializer === 'function') {
		this.update(initializer)
	}
}

neuralnet.Net.prototype = {
	update: function(func) {
		var weights = this.weights,
			length = weights.length
		
		for (var i = 0; i < length; ++i) {
			weights[i] = func(weights[i], i)
		}
	},
	updateLayer: function(n, func) {
		var weights = this.weights,
			sizes = this.sizes,
			begin = 0
			
		for (var i = 0; i < n; ++i) {
			begin += (sizes[i] + 1) * sizes[i + 1]
		}
		
		for (var i = begin, end = begin + (sizes[n] + 1) * sizes[n + 1]; i < end; ++i) {
			weights[i] = func(weights[i], i - begin)
		}
	},
	setInput: function(generator) {
		var data = this.data,
			numInputs = this.sizes[0]
		
		for (var i = 0; i < numInputs; ++i) {
			data[i] = generator(i)
		}
	},
	getOutput: function(receiver) {
		var data = this.data,
			end = data.length,
			begin = end - this.sizes[this.sizes.length - 1]
		
		for (var i = begin; i < end; ++i) {
			receiver(data[i], i - begin)
		}
	},
	getOutputAt: function(outputIndex) {
		return this.data[this.data.length - this.sizes[this.sizes.length - 1] + outputIndex]
	},
	calculate: function(activator) {
		var weights = this.weights,
			data = this.data,
			sizes = this.sizes,
			depth = sizes.length - 1,
			inputIndex = 0,
			weightIndex = 0
		
		for (var i = 0; i < depth; ++i) {
			var numInputs = sizes[i],
				numOutputs = sizes[i + 1], // num neurons
				weightsPerNeuron = numInputs + 1, // +1 for bias
				outputIndex = inputIndex + numInputs
			
			for (var j = 0; j < numOutputs; ++j) {
				var sum = 0
				
				for (var k = 0; k < numInputs; ++k) {
					sum += weights[weightIndex + k] * data[inputIndex + k]
				}
				
				sum -= weights[weightIndex + numInputs] // bias
				
				data[outputIndex + j] = activator(sum)
				
				weightIndex += weightsPerNeuron
			}
			
			inputIndex = outputIndex
		}
	},
	// requires equal sizes, only copies weights
	copy: function(net) {
		var weights = this.weights,
			netWeights = net.weights,
			length = weights.length
		
		for (var i = 0; i < length; ++i) {
			weights[i] = netWeights[i]
		}
		
		return this
	},
	clone: function() {
		return new neuralnet.Net(this.sizes).copy(this)
	}
}
