import random
import itertools
import matplotlib.pyplot as plt

import numpy as np

class Card():
	def __init__(self, num):
		self.num = num

	def __repr__(self):
		if self.num < 0:
			return f"{-self.num}: red"

		return f"{self.num}: black"

class Deck():

	def __init__(self, seed=42):
		self.cards = [Card(-i) for i in range(1, 11)] + [Card(i) for i in range(1, 11)]*2
		print(self.cards)
		random.seed(seed)

	def draw(self, start=False):

		card = random.choice(self.cards)
		# print(card)
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
		if self.winner:
			return f" Dealer's first card: {self.dealer_first}\n Player sum: {self.player_sum}\n The game is finished"

		return f" Dealer's first card: {self.dealer_first}\n Player sum: {self.player_sum}\n The game is not finished"

class Env():

	def __init__(self):
		self.deck = Deck()
		player_card = self.deck.draw(start=True)		
		dealer_card = self.deck.draw(start=True)
		self.state = State(dealer_card, player_card.num)
		self.debug = 0

	def print(self, text):
		if self.debug:
			print(text)
	def __dealer_turn(self):

		card = self.deck.draw()
		# print(f"O dealer tirou a carta {card}")
		return card

	def __dealer_loop(self):

		dealer_sum = 0
		while dealer_sum < 17:
			dealer_sum += self.__dealer_turn().num

		return dealer_sum

	def step(self, state, action):

		if action == 'hit':

			card = self.deck.draw()

			player_sum = state.player_sum + card.num
			if player_sum < 0:
				player_sum = 0

			if state.player_sum + card.num > 21 :
				s_ = State(state.dealer_first, player_sum, winner=-1)
				r = -1
				self.print("The House wins!")
				return s_, -1

			if state.player_sum == 21:
				s_ = State(state.dealer_first, player_sum, winner=1)
				return s_, 1

			s_ = State(state.dealer_first, player_sum)
			return s_, 0

		else:
			dealer_sum = self.__dealer_loop()

			if dealer_sum == state.player_sum:
				s_ = State(state.dealer_first, state.player_sum, winner=0)
				r = 0
				self.print("Draw!")

			elif dealer_sum > 21 or dealer_sum < state.player_sum:
				s_ = State(state.dealer_first, state.player_sum, winner=1)
				r = 1
				self.print("Player wins!")

			else:
				s_ = State(state.dealer_first, state.player_sum, winner=-1)
				r = -1
				self.print("The House wins!")

			return s_, r

	def sample_episode(self, pick_action):
		
			g = 0
			player_card = self.deck.draw(start=True)		
			dealer_card = self.deck.draw(start=True)
			s = State(dealer_card, player_card.num)
			a = pick_action(s)
			s_, r = self.step(s, a)
			g += g + r
			episode = [(s,a,g)]
			s = s_
			while s.winner == None:
				a = pick_action(s)
				s, r = self.step(s, a)
				g += g + r
				episode.append([s, a, g])
				# print(s)

			return episode


class Learner():
	def __init__(self):
		self.value_function = lambda s: max(self.get_q_value(s,'hit'), self.get_q_value(s, 'stick'))
		self.state_count = {}
		self.q_values = {}
		self.n0 = 100

	def get_state_count(self, state, action):

		dealer_first = state.dealer_first.num
		player_sum = state.player_sum
		try:
			n_st = self.state_count[(dealer_first, player_sum, action)]
		except:
			n_st = self.state_count[(dealer_first, player_sum, action)] = 0

		return n_st

	def get_q_value(self, state, action):

		dealer_first = state.dealer_first.num
		player_sum = state.player_sum

		try:
			q_st = self.q_values[(dealer_first, player_sum, action)]
		except:
			q_st = self.q_values[(dealer_first, player_sum, action)] = 0

		return q_st

	def get_step_size(self, state, action):

		dealer_first = state.dealer_first.num
		player_sum = state.player_sum

		return 1/self.state_count[(dealer_first, player_sum, action)]

	def update_state_count(self, state, action):

		dealer_first = state.dealer_first.num
		player_sum = state.player_sum
		try:
			self.state_count[(dealer_first, player_sum, action)] += 1
		except:
			self.state_count[(dealer_first, player_sum, action)] = 1

	def update_q_value(self, state, action, q):

		dealer_first = state.dealer_first.num
		player_sum = state.player_sum
		try:
			self.q_values[(dealer_first, player_sum, action)] += q
		except:
			self.q_values[(dealer_first, player_sum, action)] = q

	def pick_action(self, state):

		n_st = self.get_state_count(state, 'hit') + self.get_state_count(state, 'stick')
		epsilon = self.n0/(self.n0 + n_st)
		e_greedy = epsilon > random.uniform(0,1)

		if e_greedy:
			# maximize q ?
			q_hit = self.get_q_value(state, 'hit')
			q_stick = self.get_q_value(state, 'stick')

			action = 'hit' if q_hit > q_stick else 'stick'

		else:
			action = random.choice(['hit', 'stick'])

		return action

	def train(self, n_eps, env):
		# abstract method
		pass

class MCLearner(Learner):

	def __init__(self):
		super().__init__()

	def train(self, n_eps, env):

		for _ in range(n_eps):

			episode = env.sample_episode(self.pick_action)

			for s, a, g in episode:
				# print(s, a)
				self.update_state_count(s, a)
				q = self.get_q_value(s, a) + (g*self.get_step_size(s,a) - self.get_q_value(s, a))
				self.update_q_value(s, a, q)


def main():
	mc_learnr = MCLearner()
	env = Env()
	n_eps = 100000

	mc_leaner.train(n_eps, env)

	dealer_showing = list(range(1, 11))
	player_sum = list(range(1, 31))
	winners = [None, -1, 0, 1]
	axis = itertools.product(dealer_showing, player_sum, winners)

	x_axis = []
	y_axis = []
	z_axis = []

	for dealer_first, player_sum, r  in axis:

		state = State(Card(dealer_first), player_sum, winner=r)

		if player_sum < dealer_first and (r == 1 or r == 0):
			continue
		if player_sum > 21 and r != -1:
			continue 
		if player_sum == 21 and (r == -1 or r == None):
			continue 

		x_axis.append(dealer_first)
		y_axis.append(player_sum)
		z_axis.append(mc_leaner.value_function(state))

	fig = plt.figure(figsize=(4,4))
	ax = plt.axes(projection='3d')
	ax.plot(x_axis, y_axis, z_axis)
	plt.show()

main()