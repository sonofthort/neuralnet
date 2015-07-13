#pragma once

#include <cstring>
#include <iostream>

struct Connect4
{
	enum Cell {
		Red,
		Black,
		None
	};

	static const char *CellToString(Cell cell) {
		switch (cell) {
		case Red: return "Red";
		case Black: return "Black";
		case None: return "None";
		default: return "Unknown";
		}
	}

	Connect4() {
		reset();
	}

	void reset() {
		size = 0;

		each([](Cell &cell) {
			cell = Connect4::None;
		});
	}

	template <class Func>
	void each(Func func) {
		for (int y = 0; y < 8; ++y) {
			auto &cells_y = cells[y];
			for (int x = 0; x < 8; ++x) {
				func(cells_y[x]);
			}
		}
	}

	template <class Func>
	void each(Func func) const {
		for (int y = 0; y < 8; ++y) {
			auto &cells_y = cells[y];
			for (int x = 0; x < 8; ++x) {
				func(cells_y[x]);
			}
		}
	}

	bool add(Cell cell, int x) {
		if (x < 0 || x > 7) {
			return false;
		}

		int y = 0;

		for (; y < 8; ++y) {
			if (cells[y][x] != None) {
				break;
			}
		}

		if (y == 0) {
			return false;
		}

		--y;

		cells[y][x] = cell;

		++size;

		return true;
	}

	template <class Actor1, class Actor2>
	Cell automate(Actor1 actor1, Actor2 actor2, int &numTurns) {
		reset();

		const Connect4 &const_self = *this;
		numTurns = 0;

		for (;;) {
			int stall = 0;

			++numTurns;

			if (add(Cell::Red, actor1(const_self, Cell::Red))) {
				if (won() == Cell::Red) {
					return Cell::Red;
				}
			}
			else {
				++stall;
			}

			++numTurns;

			if (add(Cell::Black, actor2(const_self, Cell::Black))) {
				if (won() == Cell::Black) {
					return Cell::Black;
				}
			}
			else {
				++stall;
			}

			if (stall == 2) {
				return Cell::None;
			}
		}
	}

	Cell won() const {
		if (size == 64) {
			return None;
		}

		const Cell values[] = {Red, Black};

		for (std::size_t i = 0; i < 2; ++i) {
			const Cell value = values[i];

			for (int y = 0; y <= 4; ++y) {
				for (int x = 0; x <= 4; ++x) {
					if (cells[y][x] == value && cells[y + 1][x + 1] == value && cells[y + 2][x + 2] == value && cells[y + 3][x + 3] == value) {
						return value;
					}
				}
			}

			for (int y = 3; y < 8; ++y) {
				for (int x = 0; x <= 4; ++x) {
					if (cells[y][x] == value && cells[y - 1][x + 1] == value && cells[y - 2][x + 2] == value && cells[y - 3][x + 3] == value) {
						return value;
					}
				}
			}

			for (int y = 0; y < 8; ++y) {
				for (int x = 0; x <= 4; ++x) {
					if (cells[y][x] == value && cells[y][x + 1] == value && cells[y][x + 2] == value && cells[y][x + 3] == value) {
						return value;
					}
				}
			}

			for (int y = 0; y <= 4; ++y) {
				for (int x = 0; x < 8; ++x) {
					if (cells[y][x] == value && cells[y + 1][x] == value && cells[y + 2][x] == value && cells[y + 3][x] == value) {
						return value;
					}
				}
			}
		}

		return None;
	}

	void draw() const {
		for (int i = 1; i <= 8; ++i) {
			std::cout << i;
		}

		std::cout << '\n';

		for (int y = 0; y < 8; ++y) {
			for (int x = 0; x < 8; ++x) {
				const Cell c = cells[y][x];
				std::cout << (c == Red ? 'X' : c == Black ? 'O' : '-');
			}
			std::cout << '\n';
		}
	}

	Cell at(int x, int y) const {
		return cells[y][x];
	}

private:
	Cell cells[8][8];
	int size;
};