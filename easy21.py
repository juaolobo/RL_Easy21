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



class Learner():
	def __init__(self, gamma=1):
		self.value_function = lambda s: max(self.get_q_value(s,'hit'), self.get_q_value(s, 'stick'))
		self.state_count = {}
		self.q_values = {}
		self.n0 = 100
		self.gamma = gamma

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

	def __init__(self, gamma=1):
		super().__init__(gamma)

	def sample_episode(self, env):
		
			g = 0
			player_card = env.deck.draw(start=True)		
			dealer_card = env.deck.draw(start=True)
			s = State(dealer_card, player_card.num)
			a = self.pick_action(s)
			s_, r = env.step(s, a)
			g += g + r
			episode = [(s,a,g)]
			s = s_
			while s.winner == None:
				a = self.pick_action(s)
				s, r = env.step(s, a)
				g += g + r
				episode.append([s, a, g])
				# print(s)

			return episode

	def train(self, n_eps, env):

		for _ in range(n_eps):

			episode = self.sample_episode(env)

			for s, a, g in episode:
				# print(s, a)
				self.update_state_count(s, a)
				q = self.get_q_value(s, a) + self.gamma*(g*self.get_step_size(s,a) - self.get_q_value(s, a))
				self.update_q_value(s, a, q)


	def plot(self):
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
			z_axis.append(self.value_function(state))

		fig = plt.figure(figsize=(4,4))
		ax = plt.axes(projection='3d')
		ax.plot(x_axis, y_axis, z_axis)
		plt.show()

class TDLearner(Learner):
	def __init__(self, LAMBDA=0.9, gamma=1):
		super().__init__(gamma)
		self.state_eligibity = {}
		self.td_lambda = LAMBDA
		self.state_space = []
		self.init_state_space()

	def get_eligibity(self, state, action):

		dealer_first = state.dealer_first.num
		player_sum = state.player_sum
		try:
			elig = self.state_count[(dealer_first, player_sum, action)]
		except:
			elig = self.state_count[(dealer_first, player_sum, action)] = 0

		return elig

	def update_eligibity(self, state, action, e):

		dealer_first = state.dealer_first.num
		player_sum = state.player_sum
		try:
			self.state_count[(dealer_first, player_sum, action)] += e
		except:
			self.state_count[(dealer_first, player_sum, action)] = e

	def init_state_space(self):

		dealer_showing = list(range(1, 11))
		player_sum = list(range(1, 31))
		winners = [None, -1, 0, 1]
		product = itertools.product(dealer_showing, player_sum, winners)

		for dealer_first, player_sum, r  in product:

			state = State(Card(dealer_first), player_sum, winner=r)

			if player_sum < dealer_first and (r == 1 or r == 0):
				continue
			if player_sum > 21 and r != -1:
				continue 
			if player_sum == 21 and (r == -1 or r == None):
				continue 

			self.state_space.append(state)

	def train(self, n_eps, env):

		for _ in range(n_eps):

			player_card = env.deck.draw(start=True)		
			dealer_card = env.deck.draw(start=True)
			S = State(dealer_card, player_card.num)
			A = self.pick_action(S)

			while S.winner == None:
				S_, r = env.step(S, A)
				A_ = self.pick_action(S_)
				delta = r + self.gamma*self.get_q_value(S_, A_) - self.get_q_value(S, A)
				self.update_eligibity(S, A, 1)
				# for all states and actions update q(s,a) and e(s,a)
				alpha = 1
				for s in self.state_space:
					for a in ['hit', 'stick']:
						q = self.get_q_value(S,A) + alpha*delta*self.get_eligibity(s, a)
						e = self.gamma*self.td_lambda*self.get_eligibity(s,a)
						print(q, e)
						self.update_q_value(s, a, q)
						self.update_eligibity(s, a, e)
						print(self.get_q_value(s, a), self.get_eligibity(s, a))
				S = S_
				A = A_

def plot_mse(mc_learner, td_learners):

	mses = []
	for td_learner in td_learners:
		mse = 0
		for s in td_learner.state_space:
			for a in ['hit', 'stick']:
				mse_ = (mc_learner.get_q_value(s, a) - td_learner.get_q_value(s, a))**2
				mse += mse_
				print('to aqui', mse)

		mses.append(mse)

	y_axis = [0.1*i for i in range(0, 11)]
	fig = plt.figure(figsize=(4,4))
	ax = plt.axes()
	print(mses, y_axis)
	ax.plot(mses, y_axis)
	plt.show()


def main():
	mc_learner = MCLearner()
	td_learners = [TDLearner(LAMBDA=0.1*i) for i in range(0, 11)]

	env = Env()
	n_eps_mc = 100000

	mc_learner.train(n_eps_mc, env)
	# mc_learner.plot()
	for td_learner in td_learners:
		print('opa')
		td_learner.train(1000, env)

	plot_mse(mc_learner, td_learners)


if __name__ == '__main__':
	main()
