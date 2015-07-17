importScripts('neuralnet.js'); 

onmessage = function(e) {
	console.log('Worker started')
	
	var nn = neuralnet
	
	// create a net with one hidden layer that accepts 2 inputs, and returns 2 outputs,
	// and one outer layer, which accepts two inputs, and returns 1 output
	var best = new nn.Net([2, 2, 1], function() {
		return Math.random() * 2 - 1
	})
	
	var net = best.clone() 
	
	var maxFitness = 0
	
	// try this many mutations
	for (var n = 0; n < 10000; ++n) {
		var fitness = 0
		
		// mutate net
		net.update(function(weight) {
			return Math.random() < 0.05 ? Math.random() * 2 - 1 : weight
		})
		
		for (var i = 0; i < 10000; ++i) {
			var a = Math.random(),
				b = Math.random()
				
			net.data[0] = a
			net.data[1] = b
			
			var answer = a > b
			
			net.calculate(nn.sigmoid)
			
			var prediction = net.getOutputAt(0) >= 0.5
			
			if (prediction === answer) {
				++fitness
			}
		}
		
		// string messages are just logged to the console
		var message = n + ' ' + maxFitness
		
		if (fitness > maxFitness) {
			maxFitness = fitness
			best.copy(net)
			message += '  !!'
		} else {
			net.copy(best)
		}
		
		console.log(message)
	}
}