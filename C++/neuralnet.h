#pragma once

#include <algorithm>
#include <random>
#include <cstdint>

namespace neuralnet
{
	namespace detail
	{
		namespace random
		{
			constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();

			std::random_device device;
			std::mt19937_64 engine(device());
			std::uniform_int_distribution<std::uint64_t> distribution(0, max);

			std::uint64_t next() {
				return distribution(engine);
			}
		}
	}

	template <class T>
	T randf() {
		return static_cast<T>(detail::random::next()) / detail::random::max;
	}

	template <class T>
	T randf(T max) {
		return randf<T>() * max;
	}

	template <class T>
	T randf(T max, T min) {
		return randf(max - min) + min;
	}

	template <class T>
	struct RandDistro {
		T min, max;

		template <class...Ignore>
		T operator()(Ignore&&...) const {
			return randf(max, min);
		}
	};

	const struct sigmoid_t {
		template <class T>
		constexpr T operator()(T value) const {
			return 1 / (1 + std::exp(-value));
		}
	} sigmoid = {};

	template <class Weight, std::size_t size>
	struct Neuron
	{
		template <class Func>
		void update(Func func) {
			std::transform(weights, std::end(weights), weights, func);
			bias = func(bias);
		}

		Weight calculate(const Weight (&inputs)[size]) const
		{
			Weight result = -bias;

			for (std::size_t i = 0; i < size; ++i) {
				result += inputs[i] * weights[i];
			}

			return result;
		}

		Weight weights[size];
		Weight bias;
	};

	template <class Weight, std::size_t size, std::size_t neuronSize>
	struct Layer
	{
		typedef Neuron<Weight, neuronSize> NeuronT;

		template <class Func>
		void update(Func func) {
			for (auto &neuron : neurons) {
				neuron.update(func);
			}
		}

		void calculate(const Weight (&inputs)[neuronSize], Weight (&outputs)[size]) const
		{
			for (std::size_t i = 0; i < size; ++i) {
				outputs[i] = neurons[i].calculate(inputs);
			}
		}

		NeuronT neurons[size];
	};

	template <class Weight, std::size_t size, class T, class Test = std::enable_if_t<std::is_integral<T>::value>>
	void write(Weight (&weights)[size], T value) {
		static_assert(sizeof(T) * 8 == size, "T num bits must equal num weights");

		for (std::size_t i = 0; i < size; ++i) {
			weights[i] = static_cast<Weight>((value >> i) & 1);
		}
	}

	template <class Weight, std::size_t size, class T, class Test = std::enable_if_t<std::is_integral<T>::value>>
	void read(const Weight (&weights)[size], T &value) {
		static_assert(sizeof(T) * 8 == size, "T num bits must equal num weights");

		value = 0;

		for (std::size_t i = 0; i < size; ++i) {
			value |= static_cast<T>(1 << (weights[i] >= Weight(0.5) ? 1 : 0));
		}
	}

	template <class Weight, std::size_t depth, std::size_t size, std::size_t outputSize>
	struct Net
	{
		typedef Layer<Weight, size, size> LayerT;
		typedef Layer<Weight, outputSize, size> OutputLayerT;

		LayerT hiddenLayers[depth];
		OutputLayerT outputLayer;

		template <class Func>
		void update(Func func) {
			for (auto &layer : hiddenLayers) {
				layer.update(func);
			}

			outputLayer.update(func);
		}

		template <class Activator>
		void calculate(const Weight (&inputs)[size], Weight (&outputs)[outputSize], Activator activator) const
		{
			typedef Weight InputT[size];

			InputT data1, data2;

			hiddenLayers[0].calculate(inputs, data1);

			InputT *d1 = &data1, *d2 = &data2;

			for (std::size_t i = 1; i < depth; ++i) {
				hiddenLayers[i].calculate(*d1, *d2);
				std::transform(*d2, *d2 + size, *d2, activator);
				std::swap(d1, d2);
			}

			outputLayer.calculate(*d1, outputs);

			std::transform(outputs, std::end(outputs), outputs, activator);
		}
	};
}

