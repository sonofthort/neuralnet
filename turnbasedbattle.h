#pragma once

#include <type_traits>
#include <functional>
#include <algorithm>
#include <string>

namespace turnbasedbattle
{
	struct Action;

	struct Player
	{
		float health;
		float energy;
		const Action *lastAction;
	};

	typedef const Player &PlayerConstRef;
	typedef Player &PlayerRef;

	struct Action
	{
		std::string name, description;
		std::function<bool(PlayerConstRef caster, PlayerConstRef target)> predicate;
		std::function<void(PlayerRef caster, PlayerRef target)> perform;
	};

	const Action action_none{
		"None",
		"Internal default action",
		[](PlayerConstRef caster, PlayerConstRef target) {return true;},
		[](PlayerRef caster, PlayerRef target) {}
	};

	void heal(PlayerRef target, float amount) {
		target.health = std::min(target.health + amount, 1.0f);
	}

	void battery(PlayerRef target, float amount) {
		target.energy = std::min(target.energy + amount, 1.0f);
	}

	void apply_damage(PlayerRef caster, PlayerRef target, float amount) {
		if (target.lastAction->name == "Block") {
			amount *= 0.666666666666f;
		} else if (target.lastAction->name == "Reflect") {
			caster.health -= amount;

			return;
		} else if (target.lastAction->name == "Absorb") {
			battery(target, amount / 2);

			return;
		} else if (target.lastAction->name == "Reverse") {
			heal(target, amount / 2);

			return;
		}

		target.health -= amount;
	}

	template <class Predicate, class Perform>
	Action make_energy_cost_spell(const std::string &name, std::string description, float cost, Predicate predicate, Perform perform) {
		description += std::string(" (energy cost: ");
		description += std::to_string(cost);
		description += ')';

		return Action{
			name,
			description,
			[cost, predicate](PlayerConstRef caster, PlayerConstRef target) {
				return caster.energy >= cost && predicate(caster, target);
			},
			[cost, perform](PlayerRef caster, PlayerRef target) {
				caster.energy -= cost;
				perform(caster, target);
			}
		};
	}

	template <class Perform>
	Action make_energy_cost_spell(const std::string &name, const std::string &description, float cost, Perform perform) {
		return make_energy_cost_spell(
			name,
			description,
			cost,
			[](PlayerConstRef caster, PlayerConstRef target) {return true;},
			perform
		);
	}

	const Action actions[] = {
		Action{
			"Block",
			"Reduce damage by 33%",
			[](PlayerConstRef caster, PlayerConstRef target) {return true;},
			[](PlayerRef caster, PlayerRef target) {}
		},
		Action{
			"Meditate",
			"Gain 0.0625 energy",
			[](PlayerConstRef caster, PlayerConstRef target) {
				return caster.energy < 1.0f;
			},
			[](PlayerRef caster, PlayerRef target) {
				battery(caster, 0.0625f);
			}
		},
		make_energy_cost_spell(
			"Heal",
			"Gain 0.09375 health",
			0.125f,
			[](PlayerConstRef caster, PlayerConstRef target) {
				return caster.health < 1.0f;
			},
			[](PlayerRef caster, PlayerRef target) {
				heal(caster, 0.09375f);
			}
		),
		make_energy_cost_spell(
			"Minor Damage",
			"Deal 0.125 damage",
			0.125f,
			[](PlayerRef caster, PlayerRef target) {
				apply_damage(caster, target, 0.125f);
			}
		),
		make_energy_cost_spell(
			"Major Damage",
			"Deal 0.25 damage",
			0.25f,
			[](PlayerRef caster, PlayerRef target) {
				apply_damage(caster, target, 0.25f);
			}
		),
		make_energy_cost_spell(
			"Reflect",
			"Reflect damage back to enemy",
			0.125f,
			[](PlayerRef caster, PlayerRef target) {}
		),
		make_energy_cost_spell(
			"Absorb",
			"Absorb damage as energy",
			0.125f,
			[](PlayerRef caster, PlayerRef target) {}
		),
		make_energy_cost_spell(
			"Reverse",
			"Reverse damage as health",
			0.125f,
			[](PlayerRef caster, PlayerRef target) {}
		),
		make_energy_cost_spell(
			"Copy",
			"Copy enemy's spell",
			0.125f,
			[](PlayerRef caster, PlayerRef target) {
				if (target.lastAction->name != "Copy") {
					target.lastAction->perform(caster, target);
				}
			}
		),
	};

	template <class T, std::size_t size>
	constexpr std::size_t array_size(const T(&)[size]) {
		return size;
	}

	struct Game
	{
		Game() {
			reset();
		}

		void reset() {
			std::for_each(players, std::end(players), [](PlayerRef player) {
				player.health = 1.0f;
				player.energy = 1.0f;
				player.lastAction = &action_none;
			});
		}

		void move(const Action &player1Action, const Action &player2Action) {
			const Action * const actions[] = {&player1Action, &player2Action};

			for (int i = 0; i < 2; ++i) {
				players[i].lastAction =
					actions[i]->predicate(players[i], players[1 - i]) ?
					actions[i] :
					&action_none;
			}

			for (int i = 0; i < 2; ++i) {
				players[i].lastAction->perform(players[i], players[1 - i]);
			}

			for (int i = 0; i < 2; ++i) {
				battery(players[i], 0.0625f);
			}
		}

		bool is_game_on() const {
			return players[0].health > 0 && players[1].health > 0;
		}

		bool is_game_over() const {
			return players[0].health <= 0 || players[1].health <= 0;
		}

		bool did_player_win(int playerNum) const {
			return players[1 - playerNum].health <= 0 && players[playerNum].health > players[1 - playerNum].health;
		}

		bool is_tie() const {
			return players[0].health <= 0 && players[0].health == players[1].health;
		}

		template <class Actor1, class Actor2>
		int automate(Actor1 actor1, Actor2 actor2, int maxMoves = std::numeric_limits<int>::max()) {
			reset();

			int numMoves = 0;

			do {
				move(actor1(players[0], players[1]), actor2(players[1], players[0]));
				++numMoves;
			} while (is_game_on() && numMoves < maxMoves);

			return numMoves;
		}

		Player players[2];
	};
}
