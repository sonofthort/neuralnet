#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cassert>
#include <omp.h>

#include "neuralnet.h"

namespace nn = neuralnet;

template <class T>
T prompt(const char *header) {
	T value;

	for (;;) {
		std::cout << header << ": ";
		std::cin >> value;

		if (std::cin.fail()) {
			std::cin.clear();
			std::cin.ignore();
			std::cout << "Invalid, try again.\n";
		}
		else {
			break;
		}
	}

	return value;
}

const struct ignore_t {
	void operator()(...) const {}
} ignore = {};

#include <map>
#include <set>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cctype>

// an example single-threaded genetic algo
template <
	std::size_t depth,
	std::size_t input_size,
	std::size_t output_size,
	class T,
	class Generator,
	class FitnessFunc>
nn::Net<float, depth, input_size, output_size>
algo(std::size_t evolutions, std::size_t data_size, Generator generator, FitnessFunc fitnessFunc) {
	typedef float Weight;
	typedef nn::Net<Weight, depth, input_size, output_size> NetT;

	NetT net;

	net.update(nn::RandDistro<Weight>{-1, 1});

	Weight input[input_size];
	Weight output[output_size];

	float bestFitness = 0;

	NetT best = net;

	for (std::size_t evolution = 0; evolution < evolutions; ++evolution) {
		float fitness = 0;

		for (std::size_t i = 0; i < data_size; ++i) {
			const T value = generator();

			nn::write(input, value);
			net.calculate(input, output, nn::sigmoid);

			fitness += fitnessFunc(value, output);
		}

		// make sure net == best, so that we can evolve best into net with net.update
		if (fitness > bestFitness) {
			best = net;
			bestFitness = fitness;
		} else {
			net = best;
		}

		net.update([](Weight weight) {
			return nn::randf<float>() < 0.05f ? nn::randf(Weight(1), Weight(-1)) : weight;
		});

		std::cout << evolution << ". " << bestFitness << ", " << fitness << '\n';
	}

	return best;
}

void math_test() {
	algo<1, 32, 1, int>(1000, 10000, []() {
		return std::rand();
	}, [](int value, const float (&output)[1]) -> float {
		const bool prediction = output[0] > 0.5f;
		const bool answer = value % 4 == 0;

		return prediction == answer ? 1.0f : 0.0f;
	});
}

#include "connect4.h"

void connect4_test() {
	typedef float Weight;

	static const std::size_t input_size = 64;
	static const std::size_t output_size = 8;

	static const std::size_t depth = 2;

	typedef nn::Net<Weight, depth, input_size, output_size> NetT;

	NetT net;

	const std::size_t evolutions = 10000;

	std::vector<NetT> contenders(2048);

	for (auto &contender : contenders) {
		contender.update(nn::RandDistro<Weight>{-1, 1});
	}

	auto make_nn_player = [](const NetT &net) {
		return [&](const Connect4 &game, Connect4::Cell my_color) -> int {
			Weight input[input_size];
			Weight output[output_size];

			int data_i = 0;

			game.each([&](Connect4::Cell cell) {
				input[data_i] =
					cell == Connect4::None ? 0.0f :
					cell == my_color ? 1.0f :
					2.0f;

				++data_i;
			});

			net.calculate(input, output, nn::sigmoid);

			int x = -1;

			Weight maxWeight = -1;

			for (std::size_t i = 0; i < output_size; ++i) {
				if (output[i] > maxWeight && game.at(static_cast<int>(i), 0) == Connect4::None) {
					x = static_cast<int>(i);
					maxWeight = output[i];
				}
			}

			return x;
		};
	};

	std::size_t parentIndex = 0;

	const int maxPoints = (int)contenders.size() * 2;
	const int scoreToBeat = 750 * maxPoints / 1000;

	int nextContenderIndex = 0;

	std::vector<Connect4> boards;

	#pragma omp parallel
		#pragma omp master
		{
			boards.resize(omp_get_num_threads());
		}

	for (std::size_t evolution = 0; evolution < evolutions; ++evolution) {
		net = contenders[parentIndex];

		++parentIndex;

		if (parentIndex == contenders.size()) {
			parentIndex = 0;
		}

		net.update([](Weight weight) {
			return nn::randf<float>() < 0.05f * depth ? nn::randf<Weight>(Weight(1), Weight(-1)) : weight;
		});

		int score = 0;
		int numTurns = 0;

		const int numContenders = (int)contenders.size();

		const int bestIndex = (nextContenderIndex > 0 ? nextContenderIndex : numContenders) - 1;


		#pragma omp parallel for reduction(+:score, numTurns)
		for (int i = 0; i < numContenders; ++i) {
			Connect4 &board = boards[omp_get_thread_num()];
			int n;

			if (Connect4::Red == board.automate(make_nn_player(net), make_nn_player(contenders[i]), n)) {
				++score;
			}

			numTurns += n;

			if (Connect4::Black == board.automate(make_nn_player(contenders[i]), make_nn_player(net), n)) {
				++score;
			}

			numTurns += n;
		}

		// average
		numTurns /= numContenders * 2;

		if (score > scoreToBeat) {
			contenders[nextContenderIndex] = net;

			++nextContenderIndex;

			if (nextContenderIndex >= numContenders) {
				nextContenderIndex = 0;
			}
			std::cout << "numTurns " << numTurns << " evo " << evolution << "     !\n";
		} else {
			std::cout << "numTurns " << numTurns << " evo " << evolution << "\n";
		}
	}

	Connect4::Cell turn = Connect4::Red, playerTurn = turn;

	// play the best
	const int contenderIndex = nextContenderIndex > 0 ? nextContenderIndex - 1 : (int)contenders.size() - 1;

	Connect4 game;

	auto human_player = [](const Connect4 &game, Connect4::Cell color) -> int {
		game.draw();
		return prompt<int>("Choice") - 1;
	};

	for (;;) {
		int n;

		auto result = game.automate(human_player, make_nn_player(contenders[contenderIndex]), n);
		game.draw();
		std::cout << Connect4::CellToString(result) << " won!\n";

		result = game.automate(make_nn_player(contenders[contenderIndex]), human_player, n);
		game.draw();
		std::cout << Connect4::CellToString(result) << " won!\n";
	}
}

#include "turnbasedbattle.h"

void turnbasedbattle_human_vs_human() {
	using namespace turnbasedbattle;

	Game game;

	auto humanPlayer = [](PlayerConstRef self, PlayerConstRef enemy) -> const Action & {
		std::cout << "self: health: " << self.health << " energy: " << self.energy << " last: " << self.lastAction->name << '\n';
		std::cout << "enemy: health: " << enemy.health << " energy: " << enemy.energy << " last: " << enemy.lastAction->name << '\n';

		for (std::size_t i = 0; i < array_size(actions); ++i) {
			std::cout << i << ". " << actions[i].name << '\n';
		}

		int action;

		for (;;) {
			action = prompt<int>("Choice");
			std::cin >> action;

			if (action >= 0 && action < array_size(actions)) {
				if (actions[action].predicate(self, enemy)) {
					break;
				}
			}

			std::cout << "Invalid action, try again.\n";
		}

		return actions[action];
	};

	for (;;) {
		game.automate(humanPlayer, humanPlayer);

		if (game.did_player_win(0)) {
			std::cout << "player 1 won!\n";
		} else if (game.did_player_win(1)) {
			std::cout << "player 2 won!\n";
		} else {
			std::cout << "nobody won :(\n";
		}
	}
}

void turnbasedbattle_test() {
	namespace nn = neuralnet;
	namespace tb = turnbasedbattle;

	typedef float Weight;

	static const std::size_t input_size = 6;
	static const std::size_t output_size = tb::array_size(tb::actions);

	static const std::size_t depth = 1;

	typedef nn::Net<Weight, depth, input_size, output_size> NetT;

	NetT net;

	const std::size_t evolutions = 10000;

	static const std::size_t numContenders = 2048;

	std::vector<NetT> contenders(numContenders);

	for (auto &contender : contenders) {
		contender.update(nn::RandDistro<Weight>{-1, 1});
	}

	auto get_nn_player = [](const NetT &net) {
		return[&](tb::PlayerConstRef self, tb::PlayerConstRef enemy) -> const tb::Action & {
			Weight input[input_size];

			input[0] = self.health;
			input[1] = self.energy;
			input[2] = self.lastAction == &tb::action_none ? 0.0f : self.lastAction - tb::actions + 1.0f;
			input[3] = enemy.health;
			input[4] = enemy.energy;
			input[5] = enemy.lastAction == &tb::action_none ? 0.0f : enemy.lastAction - tb::actions + 1.0f;

			Weight output[output_size];

			net.calculate(input, output, nn::sigmoid);

			std::size_t max_index = 0;

			for (std::size_t i = 1; i < output_size; ++i) {
				if (output[i] > output[max_index] && tb::actions[i].predicate(self, enemy)) {
					max_index = i;
				}
			}

			return tb::actions[max_index];
		};
	};

	// returns 0 = no wins, 1 = a wins, 2 = b wins
	auto compete = [get_nn_player](const NetT &a, const NetT &b, int &numTurns) -> int {
		auto p1 = get_nn_player(a);
		auto p2 = get_nn_player(b);

		tb::Game game;

		numTurns = game.automate(p1, p2, 96);

		return
			game.did_player_win(0) ? 1 :
			game.did_player_win(1) ? 2 :
			0;
	};

	std::size_t parentIndex = 0;

	const int maxPoints = numContenders;
	const int scoreToBeat = 618 * maxPoints / 1000;
	//const int scoreToBeat = 75 * maxPoints / 100;

	int nextContenderIndex = 0;

	for (std::size_t evolution = 0; evolution < evolutions; ++evolution) {
		net = contenders[parentIndex];

		++parentIndex;

		if (parentIndex == numContenders) {
			parentIndex = 0;
		}

		net.update([](Weight weight) {
			return nn::randf<Weight>() < Weight(0.05) * depth ? nn::randf<Weight>(Weight(1), Weight(-1)) : weight;
		});

		int score = 0, numTurns = 0;

		#pragma omp parallel for reduction(+:score, numTurns)
		for (int i = 0; i < numContenders; ++i) {
			int n;

			if (1 == compete(net, contenders[i], n)) {
				++score;
			}

			numTurns += n;
		}

		numTurns /= numContenders;

		if (score > scoreToBeat) {
			contenders[nextContenderIndex] = net;

			++nextContenderIndex;

			if (nextContenderIndex >= numContenders) {
				nextContenderIndex = 0;
			}

			std::cout << "numTurns " << numTurns << " evo " << evolution << "    !\n";
		}
		else {
			std::cout << "numTurns " << numTurns << " evo " << evolution << "\n";
		}
	}

	auto humanPlayer = [](tb::PlayerConstRef self, tb::PlayerConstRef enemy) -> const tb::Action & {
		std::cout << "self: health: " << self.health << " energy: " << self.energy << " last: " << self.lastAction->name << '\n';
		std::cout << "enemy: health: " << enemy.health << " energy: " << enemy.energy << " last: " << enemy.lastAction->name << '\n';

		for (std::size_t i = 0; i < tb::array_size(tb::actions); ++i) {
			std::cout << "  " << i + 1 << ". " << tb::actions[i].name << "\n     - " << tb::actions[i].description << '\n';
		}

		int action;

		for (;;) {
			action = prompt<int>("Choice") - 1;

			if (action >= 0 && action < tb::array_size(tb::actions)) {
				if (tb::actions[action].predicate(self, enemy)) {
					break;
				}
			}

			std::cout << "Invalid action, try again.\n";
		}

		return tb::actions[action];
	};

	auto aiPlayer = get_nn_player(contenders[(nextContenderIndex > 0 ? nextContenderIndex  : numContenders) - 1]);

	for (;;) {
		tb::Game game;

		game.automate(humanPlayer, aiPlayer);

		if (game.did_player_win(0)) {
			std::cout << "player 1 won!\n";
		}
		else if (game.did_player_win(1)) {
			std::cout << "player 2 won!\n";
		}
		else {
			std::cout << "nobody won :(\n";
		}
	}
}

int main()
{
	srand((unsigned int)time(NULL));

	//math_test();

	//connect4_test();

	turnbasedbattle_test();
	
	return 0;
}