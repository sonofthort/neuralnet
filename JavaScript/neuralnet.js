var neuralnet = {}

neuralnet.fastexp3 = function(x) {
    return (6 + x * (6 + x * (3 + x))) * 0.16666666
}

neuralnet.fastexp4 = function(x) {
    return (24 + x * (24 + x * (12 + x * (4 + x)))) * 0.041666666
}

neuralnet.fastexp5 = function(x) {
    return (120 + x * (120 + x * (60 + x * (20 + x * (5 + x))))) * 0.0083333333
}

neuralnet.fastexp6 = function(x) {
    return (720 + x * (720 + x * (360 + x * (120 + x * (30 + x * (6 + x)))))) * 0.0013888888
}

neuralnet.fastexp7 = function(x) {
    return (5040 + x * (5040 + x * (2520 + x * (840 + x * (210 + x * (42 + x * (7 + x))))))) * 0.00019841269
}

neuralnet.fastexp8 = function(x) {
    return (40320 + x * (40320 + x * (20160 + x * (6720 + x * (1680 + x * (336 + x * (56 + x * (8 + x)))))))) * 2.4801587301e-5
}

neuralnet.fastexp9 = function(x) {
  return (362880 + x * (362880 + x * (181440 + x * (60480 + x * (15120 + x * (3024 + x * (504 + x * (72 + x * (9 + x))))))))) * 2.75573192e-6
}

neuralnet.sigmoid = function(x) {
	return 1 / (1 + neuralnet.fastexp3(-x))
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
