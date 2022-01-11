import random

class Card():
	def __init__(self, num):
		self.num = num

	def __repr__(self):
		if self.num < 0:
			return f"{-self.num}: red"

		return f"{self.num}: black"

class Deck():

	def __init__(self, seed=42):
		self.cards = [Card(-i) for i in range(10)] + [Card(i) for i in range(10)]*2
		random.seed(seed)

	def draw(self, start=False):
		card = random.choice(self.cards)

		if start:
			return Card(abs(card.num))

		return card

class State():

	def __init__(self, *args, winner=None):

		self.winner = winner
		if not args:
			self.dealer_first = None
			self.player_sum = 0

		else:
			self.dealer_first, self.player_sum = args

	def __repr__(self):
		if self.terminal:
			return f" Dealer's first card: {self.dealer_first}\n Player sum: {self.player_sum}\n The game is finished"

		return f" Dealer's first card: {self.dealer_first}\n Player sum: {self.player_sum}\n The game is not finished"

class Env():

	def __init__(self):
		self.deck = Deck()
		player_card = self.deck.draw(start=True)		
		dealer_card = self.deck.draw(start=True)
		self.state = State(dealer_card, player_card.num, False)

	def __dealer_turn(self):

		card = self.deck.draw()
		print(f"O dealer tirou a carta {card}")
		return card

	def __dealer_loop(self):

		dealer_sum = 0
		while dealer_sum < 17:
			dealer_sum += self.__dealer_turn().num

		return dealer_sum

	def step(self, state, action):

		if action == 'hit':

			card = self.deck.draw()
			s_ = State(state.dealer_first, state.player_sum + card.num)

			return s_, 0

		else:
			dealer_sum = self.__dealer_loop()

			if dealer_sum == self.state.player_sum:
				s_ = State(state.dealer_first, state.player_sum, winner=0)
				r = 0
				print("Draw!")

			elif dealer_sum > 21 or dealer_sum < self.state.player_sum:
				s_ = State(state.dealer_first, state.player_sum, winner=1)
				r = 1
				print("Player wins!")

			else:
				s_ = State(state.dealer_first, state.player_sum, winner=-1)
				r = -1
				print("The House wins!")

			return s, r
