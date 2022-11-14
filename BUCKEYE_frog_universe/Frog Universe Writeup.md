Hippity-Hop My Frog's Gone to Prague and is Going To Make Me A Dog 

# Frog Universe
## by Aryan Garg 

## Introduction

Problem Statement: 

```
Welcome to Frog Universe! Can you wander to the flag and back to _actually_ receive it? If you encounter a frog or nebula, it's game over. Thankfully, frogs will 'ribbit,' 'giggle,' and 'chirp,' and nebulas will 'light,' 'dust,' and 'dense.'
```

And the provided code looks like the following:

```python 
import numpy

import random

M = 2034

directions = {'a':numpy.array([0, -1]), 'w':numpy.array([-1, 0]), 's':numpy.array([1, 0]), 'd':numpy.array([0, 1])}

conditions = {'normal':0, 'frog':1, 'nebula':3, 'flag':5}

frog_warnings = ['ribbit', 'giggle', 'chirp']

nebula_warnings = ['light', 'dust', 'dense']

frog_loss = ['slurp', 'ribbity!', 'the frog...']

nebula_loss = ['everything is light', 'it is crushing', 'intense heat']

class Maze:

    def __init__(self, n):
        self.has_flag = False
        self.is_alive = True
        self.at_exit = True
        self.loss = ''
        self.dimension = n
        self.create_maze()
        self.position = numpy.array([self.dimension-1, 0])
        self.status = []

    def insert_flag(self):
        xi = 0
        yi = self.dimension
        while xi < self.dimension / 2:
            xi = random.randint(0, self.dimension - 1)
        while yi > self.dimension / 2:
            yi = random.randint(0, self.dimension - 1)

        print(yi, xi)
        self.maze[yi][xi] = conditions['flag']

    def assign_to_square(self, x, y):
        xi = random.randint(0, 2)
        yi = random.randint(0, 2)
        if (random.randint(0, 5)) > 1:
            self.maze[y+yi][x+xi] = conditions['frog']
        else:
            self.maze[y+yi][x+xi] = conditions['nebula']

    def insert_frogs_nebulas(self):
        for i in range(0, self.dimension-1, 3):
            for j in range(0, self.dimension-1, 3):
                self.assign_to_square(i, j)

        self.maze[self.dimension-1][0] = conditions['normal']

    def create_maze(self):
        self.maze = numpy.zeros(shape=(self.dimension, self.dimension), dtype=int)
        self.insert_frogs_nebulas()
        self.insert_flag()

    def check(self):
        warnings = []

        if self.maze[self.position[0]][self.position[1]] == conditions['frog']:
            self.is_alive = False
            self.loss = frog_loss[random.randint(0, len(frog_loss) - 1)]

        if self.maze[self.position[0]][self.position[1]] == conditions['nebula']:
            self.is_alive = False
            self.loss = nebula_loss[random.randint(0, len(nebula_loss) - 1)]
  
        if self.maze[self.position[0]][self.position[1]] == conditions['flag']:
            self.has_flag = True
            warnings.append('flag found')

        for x in directions.keys():
            temp = self.position + directions[x]
            if self.valid(temp) and self.maze[temp[0]][temp[1]] == conditions['frog']:
                warnings.append(frog_warnings[random.randint(0, len(frog_warnings) - 1)])
            if self.valid(temp) and self.maze[temp[0]][temp[1]] == conditions['nebula']:
                warnings.append(nebula_warnings[random.randint(0, len(nebula_warnings) - 1)])
        return warnings

    def valid(self, temp_position):
        return temp_position[0] > -1 and temp_position[0] < self.dimension and temp_position[1] > -1 and temp_position[1] < self.dimension

    def move(self, instruction):
        temp = self.position + directions[instruction]

        if self.valid(temp) == True:
            self.position = temp

        self.at_exit = (self.position == numpy.array([self.dimension-1, 0])).all()
        print(self.position)
        return self.check()

m = Maze(M)

print(m.position)

while m.is_alive == True and not (m.has_flag == True and m.at_exit == True):
    warnings = m.move(input())
    for x in warnings:
        print(x)

if not m.is_alive:
    print('something happened: ' + m.loss)
    print('try again!')

if m.has_flag:
    with open("flag.txt", "r") as fi:
        print(fi.read())
```

Let's Go!

## Understanding the code

When we run the provided python file, it allows us to enter keys "w", "a'", "s", "d"

It seems to first give the coordinate of our "flag", then the current coordinate of where we start (which starts as (2033,0)) 

```
456 1422
[2033    0]
w
[2032    0]
w
[2031    0]
d
[2031    1]
d
[2031    2]
w
[2030    2]
d
[2030    3]
w
[2029    3]
giggle
w
[2028    3]
something happened: ribbity!
try again!
```

It seems like we can traverse the maze using (as every gamer will know by heart) WASD

Every time we get close to an obstacle, it gives us a warning if our current player's position is adjacent to an obstacle. The warning strings are in the _warnings array

```python
frog_warnings = ['ribbit', 'giggle', 'chirp']
nebula_warnings = ['light', 'dust', 'dense']
```


Looking into our main game loop:
```python
while m.is_alive == True and not (m.has_flag == True and m.at_exit == True):
```

It seems like what we would have to do is that we have to start at (2033, 0), make it to whatever (x_flag, y_flag) they give us in the beginning, then make it back to (2033,0) without running into a bomb and dying.

Hmm... Ok, but now where do we start?

## Similarities with other problems in Computer Science

When first messing around with the game and figuring out how to do this, the first thing I thought of was Minesweeper:

![[minesweeper.gif]]

We are essentially on one huge grid and we have to figure out how to explore the grid without running into a bomb, and we are given the information on how many bombs are around us at each step we take. The difference between this game and our game though is that we are not given a whole grid to work with, but instead just a single point in space.

This game also reminds me of another type of Computer Science problem, Maze Solving. The main question is maze solving consists of finding a path from point A to point B given a set of obstacles / nodes and edges. The difference between our game and maze solving though is that we are not given where our obstacles are, all that we know is whether or not we are near a bomb.

## Solving Minesweeper

First, we will look at a Minesweeper. Given a Minesweeper board, how will we complete a game without dying? 

Note: In Minesweeper, warnings are given to us by looking at all 9 of a grids neighbors. The game we are working with only tells if there are any bombs in any of its 4 adjacent neighbors.  

Well the first thing that happens when you start a minesweeper game, you will click somewhere and uncover all the squares around you which are guaranteed to be safe. 

![[minesweeper_uncover.gif]]

So when we click on a single grid, we will check if this grid is a bomb or not. If it is a 0, then we can also further figure out that all adjacent grids to our initial grid will also not have a bomb in it. So we will be able to safely reveal them.
We can then check each of those grids and see if they also have a 0, in which case we would know that their neighbors are also safe to visit.
Continuing this process, we can find a recursive-like algorithm to uncover all out safe points in our gameboard, all the way until we get to the "edge" of the gameboard where only non-zero tiles exist. We will call this method of revealing tiles the "0-neighbor" method. 
I will go over how we can implement this "recursive-ish" way to reveal squares in the next section, but first I am going to discuss what to do next after we run out of tiles to find using the "0-neighbor" method and only our "edge" of non-zero tiles exists. 

### Discovering new safe points using Neighbor Pruning 

So now let's assume we are at the point where we chose our initial position, we have been able to see all the adjacent points to that point and all the adjacent points to the adjacent points of that point (and so on...) that are safe. We are now at the point where we do not have any more places where we can easily guarantee that it will not have a bomb (because the point does not have a 0 square next to it telling us that). What can we do?

Well, there are multiple techniques in figuring out what to do next. One useful technique is to try and think like how a human will try and decipher playing the game and how they would know whether a point is safe or not, and see how we can turn that into a computational way of thinking.

![[minesweeper_one_neighbor.png]]

One case we may run into is like the one above. Look at the 1 in the middle. The 1 tells us that there is only one bomb that surrounds it. If we look at its neighbors, we know 8 of that squares neighbors and 1 of the squares neighbors is unknown. Since there is one neighbor available for us to pick from and one bomb that surrounds the bomb, we can say that square has a bomb.

Now we can mark that position as a bomb and we can check all the neighbors of that bomb to see if there were any other squares we might be able to deduce because of this new information


![[minesweeper_one_neighbor_marked.png | 300]]

(I will be referring to the coordinates of these grids (y,x) where (1,1) is the top left and (5,5) is the bottom right)

Since we know the position of the bomb is (2,3), so we can check all 8 of its adjacent points

(2,4), (1,4), (1,3), (1,2)

These points are still unknown to us so we don't touch it / change much 

(3,2), (3,3), (3,4)
These points are all marked as a 1, and we now know the location of the bomb that that warning is for1. This means that every other coordinate surrounding the 1's are not a bomb, and we can mark all these other adjacent points as safe. We will mark points (2,4) and (2,5) as safe, and we may also add these cords to some type of list to keep track of possible points we can use to recursively discover using our 0-neighbor pruning trick later on.

(2,2)

Since we now know new information about the bombs surrounding this point, we re-check this point. We now know one of the two bombs that surround this point, but since we still do not know the second bomb we can't do anything more with this point.


There are some more techniques we can use to try and discover new safe spots, but it turns out the only two we really need to solve our flag is the Zero-Neighbor and our Neighbor-Pruning technique. 
So now our next question is how exactly can we implement this "recursive" discovery algorithm I talked about earlier? Well, first we will have to dive deep into the realm of Graph Traversal in Computer Science.


## Graph Traversal

What exactly is a graph?
Well in loose terms it is simply a collection of nodes, of which they are connected by edges.

![[graph.png]]

One common problem in computer science is: Given two nodes, how can we find a pathway from one node to the other?

For example, If I give you node 3, can you tell me the steps I would need to take from getting to point 3 to point 6? 
	(In this example one set of steps could be: 3 -> 5 -> 4 -> 6)

One solution to this problem that computer scientists have found is an algorithm called Breadth-First Search. 

![[bfs.gif | 500]]

How it works is that it keeps a queue of all points that it wants to visit next. The queue (which you can think of as a real-life queue of people, Last in First out) is first initialized to our starting point, and all the children of that point will be added to the queue. After that, it gets the first node out of the queue, checks if the current node is our goal_node, and if not it adds all of its children into the queue. That cycle keeps on going on until we find the node we want. 

```python
queue = []     #Initialize a queue

def bfs(graph, init_node, goal_node): # function for BFS
  # graph = an array which tells us all the edges of a given node 
  # init_node = node we start off at 
  # goal_node = node we want to get to  
  queue.append(init_node)

  while queue:          # Creating loop to visit each node
    m = queue.pop(0) 

	if m == goal_node:
		return True

    for neighbour in graph[m]:
        queue.append(neighbour)
```

Note: The above is a very simplified version of Breadth First Search in python, and foregoes some more features that we can add to Breadth First Search, such as the following:
- A Visited set to make sure we are not re-visiting nodes that we have already seen
- A way to keep track of the path we are taking. One way we can do this is to store the path we are taking when adding a node to the queue (ex. in an object structure such as `{node: node, path: node[]}` )


### Applications to Path Finding / Minesweeper

Okay cool now we got the general premise on how we can try and "traverse" a graph, but how does this help us with Minesweeper at all?

Well, one of the first things we can do is transform what we can think of as a graph. 
We can convert our minesweeper grid into a graph but think of each grid as something we can traverse onto, and there will exist 4 edges connecting our point to each of its adjacent neighbors.

![[grid_to_graph.jpg]]

To try and get a better intuition on how a breadth-first search algorithm might work with grids, we can look at maze-solving, where given a maze / an array of walls  we would need to find the solution to our maze.

![[bfs_pathfinding.gif | 400]]

If you look at the above gif, you can see how breadth-first search is working. At each of the nodes, you add all of the neighboring nodes to a queue. You visit all of those nodes, and while visiting them you continue to add all of our neighboring nodes' neighbors to the queue. We then visit all of those nodes, all of those neighboring neighboring ..., etc. 

You can also see how a pattern emerges with breadth-first search where you tend to search almost "1 layer" at a time. Graph traversal is a huge concept in Computer Science and I will not be going in-depth with all of the different graph traversals in Computer Science, but below are some more examples of popular graph traversals and how they look.

**Depth First Search**

![[dfs.gif]]

**A* Search** 

![[astar.gif]]


Let's see if we can now try to traverse this graph with Minesweeper.

### Applying Path Finding to Minesweeper 

First, let's assume our starting point is free of all bombs and warnings (if it is not, we can always restart our minesweeper game until it is).

We will first traverse to our first spot in our board and add our top, right, left, and bottom spots to our queue (since the grid does not exist at the bottom and to the left, we will not add out-of-bound points to our queue)

We will then go to the next node in the queue. If this node is a 0, then we can safely add the top, right left, and bottom spots into our queue, and continue traversing.

If our node is not a 0, that means it is rather a 1, 2, or 3 (and hopefully not a bomb otherwise that means we died). If we are at a 1, 2, or 3, we can mark this point as a "special point." The first thing we would want to check is if we already know all the locations of the bombs that this node warns us of. So if the node warns us of 2 bombs, and we already know the locations of the 2 adjacent bombs, then we can mark the other 2 adjacent points as safe and add that to our queue. If we do not already know the location of the bombs that the node warns us of, then we will not add the neighbors to the queue (to avoid stepping into a bomb), and we will move on. We would also want to remember this point for later on (by storing it in some type of set), to possibly use for our neighbor-pruning technique from earlier.

Note: We will also be keeping track of a seen set to make sure we are not traversing through a point that we have already been to before. 

So now we can go through this cycle over and over again until we get to a point where there are no longer any more spots that we can guarantee to be safe because of our 0-neighbor trick.  

Once we run out of neighbors to check with our 0-neighbor trick, we can iterate through all the points we saved in our "special points" set and check to see if we can now find any special points that can qualify into our neighbor-pruning trick from earlier. If it does, we can mark the new points as bombs and update all of that bomb's neighbors to see if there are any points we can now guarantee to be safe. If we do, we can now add these new points to our queue and then redo our Breadth-First Search on that point, finding all the points that are guaranteed to be safe. 

### Path Finding Inception

Okay Cool! Now we have a systematic way of finding our way through our game without killing ourselves. We just need to slowly inch our way around the gameboard, checking only the 0-neighboring points, then after we run out of points to check using that method we can move on to using the neighbor-pruning points to use our 0-neighbor method once again. Rinse and repeat until we finish our board.

But there is one oversight that I did not address in the above statement. We have a general idea now of how we can slowly traverse our whole graph, but how would we traverse through the game itself? For example, let's say I'm at node (40,30), and the next point in the queue tells me to go to (15,23). If you remember earlier when demonstrating the program, all I can enter is WASD. I need to somehow calculate the correct WASD moves to get from (40,30) to (15,23) without stepping on a bomb and killing myself. 

As it turns out, we can again still use Breadth First Search to figure out where to go next. This is almost the same problem as we can into at first, given a point A and a point B and a graph G, traverse the graph G from point A and find a path to B.


### TL;DR

Okay so in summary, this is what we would need to code:

> Keep track of the entire gameboard, making sure to keep track of the different states of the board that we know so far
>> some of the different states that we should keep track of include:
>>> B = Bomb
>>> 0 = Safe Spot
>>> ? = Unknown
>>> 1 = 1 neighboring bomb
>>> 2 = 2 neighboring bombs
>>> 3 = 3 neighboring bombs

> Start at our init position (which in the game is (2033, 0), and slowly Breadth-First Search through all of the safe spots (which we are checking for using our 0-neighbor method) in our gameboard
>> In our breadth-first search we will keep track of:
>>> A queue of nodes to visit next
>>> A set of special points that we can refer to later on to use for our neighbor-pruning technique
>>> a set of seen nodes that we use to keep track of the nodes we have already visited, so as to not revisit any nodes 

>During our main BFS of the entire gameboard, we would need to calculate the WASD moves to get from one node to another
>> So we will have another helper function called `go_to_location(matrix,start,end)` which will return an array of chars consisting of 'w', 'a', 's', 'd', to which we can write to our socket connecting to the server hosting the game. 
>> We will be finding the correct pathway for us to take / the wasd moves to take using another BFS algorithm
>>> So every iteration in our overall BFS algorithm, we will be using another mini BFS algorithm to get from one point to another 
>> After we traverse to that point, the server will give us an output to which we can parse into the number of warnings that we currently have at our position. 

>After we run out of positions in our 0-neighbor trick (that is our queue of nodes to visit next is empty), we can now move on to our set of special points. We can now check if we can find any more new nodes that we can add to our queue using our neighbor-pruning technique.
>> So to implement this technique, we will iterate through every point in the special_points set and see if the number of unknown neighbors equals the number of warnings for any of our special points. If it does, we can update all of the unknown positions as bombs and recheck to see if that gives us any new information about which spots are safe.

Okay great now we know the technique, let's code it!

Wait wait wait, before we code it... let's take out some pen and paper, and leetcode interviewing style check our time-complexity first...


## O(ERROR 408: Request Timed Out)

Ok, let's do some quick math.

We have a 2000 by 2000 grid, so we have approx. 4\*10^6 different grid positions to check.
Each grid position to check might take ~5 moves to move from one position to another (the # of moves gets bigger as the amount of the grid we have explored expands).
So we will send ~ 2\*10^7 different moves to our game console.
And since we have to go to the flag and back we have to multiply this number by 2, so in total, we will have to make ~ 10^8 different moves. 

Each move takes approx. 0.01 seconds to make (and that's with perfect conditions, for you gamers that means 10ms of ping). So in total, it would take ~ 10^8 \* 10^-2 = 10^6 seconds to finish our whole traversal, which is 1000000 seconds. or 277 hours.

Sadly, this competition only lasted 48 hours, so it seems like our current algorithm might not work. 

Now like every interviewer would ask, "Can we find a more efficient solution?"

### Looking For Optimal Graph Traversals

Well one thing that might help us is looking back at the visualization of our graph traversal, and seeing if there is anything we could optimize with it. 

![[breadth_first_search_slow.gif]]

As you can see, currently our Breadth-First Search Expands out as a circle. Every single whole iteration essentially adds another layer to the shell of the area it has searched. 
While this utility might be useful in some problems, in our application it gives us a lot of wasted moves. We are making quite a lot of moves in the opposite direction of where we want to go.

It seems we would be able to speed up our time-complexity quite a bit by instead of trying to search through the entire graph all at once, we try to be selective with our choice of which node to visit next. 

One technique that we can do is to choose the closest point in our queue to our target. This will create a more targeted approach to our graph traversal instead of just spraying everywhere. 

This algorithm is called Greedy Best First Search, and here is how it looks visualized: 

![[best-first-search.gif]]

As you can see with this graph traversal algorithm, instead of it working like extending out spherical shells, it works more like a directed shotgun, with it only choosing the points that get it closer to its target. 

One interesting observation is that when it runs into a wall it does tend to use the spherical-shell-like method. The reason for this is that the Best First Search method works almost exactly like the Breadth First Search Method, but the only thing is that instead of a Queue, we use a different data structure called a Priority Queue. How the priority queue works are very similar to a queue, but when adding a new element instead of it just adding it to the back of the queue, the Priority Queue keeps itself sorted according to some "weight." So the next element to be "pop"-ed out won't be the earliest element put into it, but instead the element with the least weight. In our case, we can make our weight the Euclidian Distance between the current node and our goal node. We can calculate this with good ol' Pythagoreans Theorem, `a^2 + b^2 = c^2` 

Also another small change we will need to make is we need to do our neighbor-pruning technique at every iteration to make sure we don't "bubble back" too much and are constantly finding new possible nodes to iterate through.

This will tremendously help out with our runtime and will allow us to use a much more directed approach instead of just going throughout the entire graph. This will move the number of grid positions we have to check from 10^6 to only ~10^3, turning our 277 hours into a smaller 20 minutes.

## The Code

We finally got our whole description of the problem and solution. Now we just have to code it.

I am not going to lie, coding this was a pain to program. But that's okay, as after 5 hours of banging by head into the keyboard we finally got it to work. Below is the code that we finally were able to run to get our flag.

```python
from pwn import *
import numpy as np
from enum import Enum
from collections import deque
from heapq import heappush, heappop

## Globals:

SIZE = 2034
# declare cell enum (state)
class Cell(Enum):
	SAFE_SPACE = 0
	UNKNOWN = -1
	BOMB = -2
	ONE_BOMB = 1
	TWO_BOMBS = 2
	THREE_BOMBS = 3

# Directions to Instructions
class Directions(Enum):
	NORTH = 'w'
	EAST = 'd'
	SOUTH = 's'
	WEST = 'a'


## Util Functions:

def getAllAdjPts(coord):
	dxdy = [(0,1), (0, -1), (1,0), (-1,0)]
	adjPts = []
	for dx, dy in dxdy:
		adjPts.append((coord[0] + dx, coord[1] + dy))
	return adjPts

def distToFlag(coord):
	(y,x) = coord
	return abs(x-flag_x)+abs(y-flag_y)

def addToStack(heap,coord):
	heappush(heap,(distToFlag(coord),coord))

def range_check(position):
		return position[0] > -1 and position[0] < SIZE and position[1] > -1 and position[1] < SIZE

def print_surrounding(maze, position):
	# pass
	for i in range(-10, 10):
		for j in range(-10, 10):
			if i == 0 and j == 0:
				print('P', end=' ')
			else:
				if range_check(position + np.array([i, j])):
					charToPrint = maze[position[0]+i][position[1]+j]
					if charToPrint == -1:
						charToPrint = "?"
					if charToPrint == -2:
						charToPrint = "B"
					print(charToPrint, end=' ')
				else:
					print('x', end=' ')
		print()

## I/O Functions:
def get_user_coordinate():
	# receive response
	try:
		i = 0
		print("waiting for recv")
		time.sleep(0.01)
		i = 0
		while not r.can_recv():
			time.sleep(0.01)
			i += 1
			if i > 100:
				r.interactive()
		raw_response = r.recv()
		response = str(raw_response)[2:-3].split('\\n')
		print('receive this: ', response)
		unparsed_string = response[0]
		y, x = [int(x) for x in unparsed_string[1:-1].split()]

		return y, x, len(response)
	except:
		print('something wrong with this response: ', raw_response)
		exit()

def num_of_warnings_at_end(instructions):
	num_of_bomb = -1
	for i in instructions:
		# execute the instruction
		print("sending",i)
		r.sendline(i)
		print("about to get user coord in num_warnings_at_end")
		user_y, user_x, length_of_response = get_user_coordinate()
		num_of_bomb = length_of_response - 1
	return num_of_bomb

# r = remote('pwn.chall.pwnoh.io', 13380)
r = process(argv=["python3", "maze.py"])
# r.interactive()
# get the position of the flag
flag_pos = str(r.recvline())[2:-3].split(' ')
flag_y = int(flag_pos[0])
flag_x = int(flag_pos[1])

print('This is coordinate of flag (y, x): ', flag_y, flag_x)
get_user_coordinate()


# Main Solver
def main_solve():
	seen = set()
	matrix = np.zeros((SIZE, SIZE), dtype=int)
	matrix.fill(Cell.UNKNOWN.value)
	next_visit_stack = []
	addToStack(next_visit_stack,(2032,0))
	addToStack(next_visit_stack,(2033,1))
	special_points_set = set()
	current_cordd = (2033, 0)

	# just make two points good
	matrix[2033][0] = 0
	matrix[2033][1] = 0


	while True:
		while len(next_visit_stack) != 0:
			# get next coordinate to visit
			_, coord = heappop(next_visit_stack)

			# if we have seen this coordinate before, skip
			if coord in seen or not range_check(coord):
				continue

			# If this coord is our goal coord, then traverse to this coord and collect the flag
			# After doing that, traverse back to our starting position (2033, 0) and we should have our flag!
			if coord[0] == flag_y and coord[1] == flag_x:
				go_to_location(matrix, current_cordd, coord)
				go_to_location(matrix, coord, (2033,0))
				r.interactive()
				return

			# If this coord is a bomb, skip it.
			if matrix[coord[0]][coord[1]] == Cell.BOMB.value:
				continue

			# If neither of the above conditions are satisfied, we will traverse this coordinate
			# Mark it as visited
			seen.add(coord)

			print_surrounding(matrix, coord)

			# go to our new cordd
			numOfWarnings = go_to_location(matrix, current_cordd, coord)
			print("Number Of Warnings at above loc: " + str(numOfWarnings))
			current_cordd = coord

			# If this is a 0 Coord, all neighbors are safe to visit
			if(numOfWarnings == 0):
				matrix[coord[0]][coord[1]] = Cell.SAFE_SPACE.value
				addToStack(next_visit_stack,(coord[0] - 1, coord[1]))
				addToStack(next_visit_stack,(coord[0] + 1, coord[1]))
				addToStack(next_visit_stack,(coord[0], coord[1] + 1))
				addToStack(next_visit_stack,(coord[0], coord[1] - 1))
			# If this is not a Zero Coord, we have to be cautious with traversing any of this coord's neighbors
			# So we will not immeditly add its neighbors to our stacktovisit
			elif numOfWarnings >= 1:
				# This will check if we already know all the bombs that this coord is talking about
				for adjCoord in getAllAdjPts(coord):
					if not range_check(adjCoord):
						continue
					if matrix[adjCoord[0]][adjCoord[1]] == Cell.BOMB.value:
						numOfWarnings -= 1

				# If we do, then we can safely add all the neighbors to our stacktovisit, if not we will not
				if numOfWarnings == 0:
					matrix[coord[0]][coord[1]] = Cell.SAFE_SPACE.value
					addToStack(next_visit_stack,(coord[0] - 1, coord[1]))
					addToStack(next_visit_stack,(coord[0] + 1, coord[1]))
					addToStack(next_visit_stack,(coord[0], coord[1] + 1))
					addToStack(next_visit_stack,(coord[0], coord[1] - 1))


				# update the value we have for this coord
				matrix[coord[0]][coord[1]] = numOfWarnings
				# add this into our special points list as to check for neighbor-pruning later on
				special_points_set.add(coord)

			# Now implementing neighbor pruning technique
			for special_cord in special_points_set.copy():
				if matrix[special_cord[0]][special_cord[1]] == Cell.SAFE_SPACE.value:
					if special_cord in special_points_set:
						special_points_set.remove(special_cord)
					continue

				# we will be checking all special points to see if we can find
				# a point where the total adjacent places of unknowns = num of warnings (such that all unknowns = bombs)

				# Check # of Unknowns adjacent to this special point
				numOfUnknowns = 0
				unknownCoords = []
				for newcord in getAllAdjPts(special_cord):
					if not range_check(newcord):
						continue
					if matrix[newcord[0]][newcord[1]] == Cell.UNKNOWN.value:
						numOfUnknowns += 1
						unknownCoords.append(newcord)

				# If the number of unknowns is equal to the number of warnings, then we can safely assume that all unknowns are bombs
				if numOfUnknowns == matrix[special_cord[0]][special_cord[1]]:
					## WE KNOW WHERE OUR BOMBS ARE!!!!
					# first step is replace our matrix val
					# second step is to check all surrounding special points

					for bombCoord in unknownCoords:
						# first step
						matrix[bombCoord[0]][bombCoord[1]] = Cell.BOMB.value

						# second step (check all surrounding special points)
						for newcord in getAllAdjPts(bombCoord):
							if not range_check(newcord) or matrix[newcord[0]][newcord[1]] == Cell.UNKNOWN.value \
								or matrix[newcord[0]][newcord[1]] == Cell.BOMB.value or matrix[newcord[0]][newcord[1]] == Cell.SAFE_SPACE.value:
								continue

							# if this is a point which shows we have only one bomb,
							# then we now know the bomb that caused the warning
							# we can now safely ignore this warning and now start traversal from here.
							if matrix[newcord[0]][newcord[1]] == Cell.ONE_BOMB.value:
								matrix[newcord[0]][newcord[1]] = Cell.SAFE_SPACE.value
								addToStack(next_visit_stack,newcord)
								if newcord in seen:
									seen.remove(newcord)
								special_points_set.remove(newcord)
							else:
								matrix[newcord[0]][newcord[1]] -= 1
		if len(next_visit_stack) == 0:
			print("You have no other places in the stack to visit")
			break


# This function will find the WASD moves to get from start to end and then send those moves to the server
# It will return the number of warnings it has found at the grid it traversed to
# This works using BFS
def go_to_location(matrix, start, end):
	DIRS = [-1, 0, 1, 0, -1]
	DIR_MAP = {
		0: "w",
		1: "d",
		2: "s",
		3: "a"
	}

	queue = deque([(start, [])])
	visited = set(start)

	while queue:
		curr_loc, curr_path = queue.popleft()
		curr_y, curr_x = curr_loc

		if curr_loc == end:
			print("goToLocation ", start, end, curr_path)
			return num_of_warnings_at_end(curr_path)

		for i in range(4):
			new_y = curr_y + DIRS[i]
			new_x = curr_x + DIRS[i+1]

			if range_check((new_y, new_x)) and (int(matrix[new_y][new_x]) not in [Cell.BOMB.value, Cell.UNKNOWN.value] or (new_y, new_x) == end):
				if (new_y, new_x) not in visited:
					visited.add((new_y, new_x))
					queue.append(((new_y, new_x), curr_path + [DIR_MAP[i]]))
	print('No traversal path found')

main_solve()
```

Sample Output:

```
goToLocation  (17, 51) (17, 50) ['a']
sending a
about to get user coord in num_warnings_at_end
waiting for recv
receive this:  ['[17 50]']
Number Of Warnings at above loc: 0
? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ?
? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ?
? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ?
? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ?
? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ?
? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ? ?
? ? ? ? ? ? ? ? ? ? 1 ? ? ? ? ? ? ? ? ?
? ? ? ? ? ? ? ? ? 1 0 1 ? ? ? ? ? ? ? ?
? ? ? ? ? ? ? ? 1 0 0 0 1 ? ? ? ? ? ? ?
? ? ? ? ? ? ? B 0 0 0 0 0 1 ? ? ? ? ? ?
? ? ? ? ? ? ? ? 0 0 P B 0 0 1 ? ? ? ? ?
? ? ? ? ? ? ? ? 0 0 0 ? 0 1 ? ? ? ? ? ?
? ? ? ? ? ? ? ? 0 ? ? ? ? ? ? ? ? ? ? ?
? ? ? ? ? ? ? ? 0 ? ? ? ? ? ? ? ? ? ? ?
? ? ? ? ? ? 0 B 0 ? ? ? ? ? ? ? ? ? ? ?
? ? ? ? ? ? 0 0 0 ? ? ? ? ? ? ? ? ? ? ?
? ? ? ? ? ? 0 0 0 ? ? ? ? ? ? ? ? ? ? ?
? ? ? ? ? B 0 ? ? ? ? ? ? ? ? ? ? ? ? ?
? ? ? ? B 0 0 ? ? ? ? ? ? ? ? ? ? ? ? ?
? ? ? ? 0 0 0 ? ? ? ? ? ? ? ? ? ? ? ? ?
```

You can find the full output for the program on a sample grid of 100x100 here:
https://pastebin.com/4p8m6sk3

I highly recommend checking out the PasteBin link above and scrolling through the graph and how it works. You will be able to see the snake-like feature of the traversal as well as all the bombs and how the graph looks from the player's perspective and each point in its journey (it was also a major lifesaver for debugging).

## Conclusion

Thank you to Abi, Zi, and Sam for helping me code the program. This was a painful CTF Challenge but it was really fun, and it was satisfying to be able to finally get the flag. Graph Traversal is an incredibly useful technique in Computer Science and this challenge was a great way for me to test my Graph Theory / Programming skills.

Below is a video of when we finally were able to solve the challenge :)

![[solving_flag.mp4]]
