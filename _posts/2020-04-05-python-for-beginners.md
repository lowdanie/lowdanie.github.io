---
layout: post
title: "Python for Beginners"
date: 2020-02-08
mathjax: false
---

I recently finished teaching an introductory Python course as part of the [Youth Remote Learning](https://youthremotelearning.com/2020/) project. The intended audience was 6th-8th graders and consisted of 6 classes, each lasting 30 minutes.

The goal of the class was to quickly get the students to a point where they could design their own simple games. This post contains the syllabus that I used. There is obviously nothing new here, but I thought that the structure of the lesson plan may be useful to someone that is thinking of teaching a similar class or is learning on their own.


# Part 1 - Introduction to Python
In this section we will familiarize ourselves with the python interpreter and some fundamental programming concepts.

## The Python Interpreter
First you will need to download python from here: [https://www.python.org/downloads/](https://www.python.org/downloads/).

Now open a program called IDLE (an acronym: Integrated Development and Learning Environment) which should come as part of your python installation. You should see the following text:

```
Python 3.8.2 (v3.8.2:7b3ab5921f, Feb 24 2020, 17:
52:18) 
[Clang 6.0 (clang-600.0.57)] on darwin
Type "help", "copyright", "credits" or "license()" for more information.
>>>
```

The three greater than signs `>>>` are called the _prompt_. The window containing this text is called the _shell_. You can run commands by typing them after the prompt and then pressing ENTER. Try running the following calculation:

```
>>> 1 + 1
2
```

The program that runs a command like `1+1` is called the _interpreter_. If you got the number `2` then your interpreter is working!

## Arithmetic and Variables
Let’s play around with some other arithmetic expressions:

```
>>> 2 + 5
7
>>> 2 * 3
6
>>> 10 / 4
2.5
>>> 2 * (1.5 + 2)
7
```

It is also possible to save numbers in variables.

```
>>> height = 2
>>> width = 5
>>> area = height * width
>>> print(area)
10

>>> sum = 20
>>> average = sum / 4
>>> print(average)
5.0
```

Here we have used the `print` command to display the value of a variable.

## Strings
In addition to numbers, we can also work with text which in python are called _strings_.

```
>>> "Hello World"
'Hello World'
>>> name = "Daniel"
>>> print(name)
Daniel
>>> greeting = "Hello "
>>> greeting + name
Hello Daniel
```

What do you think will happen if you multiply a string by number like this: `"# " * 10`? The best way to find out is to try it:

```
>>> "# " * 10
'# # # # # # # # # # '
```

You can also combine numbers and strings in a `print` statement. For example:

```
>>> print("2 + 2 =", 2 + 2)
2 + 2 = 4
```

Or
```
>>> level = 4
>>> print(“You reached level”, level)
You reached level 4
```

# Part 2: Programs and User Input
## Hello World
We can collect many commands into a file called a _program_. The simplest program simply prints a message. To create a program, go to IDLE and click File > New File. This should open up a new window. Type the following line into the file:

```python
print("Hello World")
```

Then save the file with File > Save. Finally, run the program by clicking Run > Run Module. You should now see the output of the program in the shell:

```
=== RESTART: /Users/daniellowengrub/Documents/remote_teaching/hello_world.py ===
Hello World
>>> 
```

## User Input
In order for a program to be interactive we must be able to get input from the person using it. We can do this with the `input` command. To try this out, create a file with the following program:

```python
print("You have reached the security vault.")
name = input("State your name> ")
print(name, "you are cleared to enter.")
```

Now run this program and it will ask you for your name.

Here is another example of a program in which we ask the user for two numbers and show them the sum:

```python
print("Hi, welcome to the magical addition machine.")
x = int(input("Enter your first number: > "))
y = int(input("Enter your second number: > "))
result = x + y
print(x, "+", y, "=", result)
```

We used the `int` command to convert the input from a string to a whole number, also known as an _integer_.

As a final example, here is a program that creates a game character:

```python
print("It is time to create your character.")
height = input("Enter a number> ")
animal = input("Enter a type of animal> ")
print("Your character is a", height, "foot tall", animal)
```

__Exercise__: Write a program that asks the user for the height and width of a rectangle and then prints the area.  
__Exercise__: Make your own version of the game character program using different character attributes.

# Part 3 - While Loops
A _loop_ is a command that lets you run the same commands over and over until a certain condition is met. Here is a simple program that uses the `while` command to count to 10:

```python
print("Hello, for my next trick, I will now count to 10")
n = 1

while n <= 10:
    print(n)
    n = n + 1

print("Bye Bye.")
```

Try this out by typing it into a file and running it as we did before. It is important that all of the lines in your program have the same indentation as the one above.

Let’s break down what’s happening here. 
1. `n = 1` We initialize a variable called n to the value of 1.
2. `while n <= 10` This is called a _while loop_. It means that it will keep running the indented code (i.e, the two lines below it) as long as the variable `n` is less than or equal to 10.
3. `print(n)` Print the current value of n.
4. `n = n + 1` Increment n by 1. Since this is the end of the loop, we will now go back to step 2.

The loop stops once the value of the variable `n` reaches 11.

Here are come common conditions that we can put in a while loop:  
`x == y`: x is equal to y (Important: Note that there are two equal signs)  
`x != y`: x is not equal to y  
`x < y`: x is less than y  
`x > y`: x is greater than y

Here are some examples that you can try in the interpreter:
```
>>> 1 == 2
False
>>> 1 < 2
True
>>> "dog" == "cat"
False
>>> "dog" != "cat"
True
>>> "dog" == "dog"
True
>>> x = "fish"
>>> y = "bird"
>>> x == y
False
```

Here is an example of using a while loop to make a program that draws a triangle:

```python
print("TRIANGLE BUILDER")
print("================")
num_rows = int(input("Enter the number of rows> "))

row = 1
while row <= num_rows:
    print("#" * row)
    row = row + 1

print()
print("Triangle complete.")
```

__Exercise__: Write a program that adds up the numbers from 1 to 100 and prints the result. I.e, calculate: 1 + 2 + 3 + … + 99 + 100. _Bonus_: Ask the user for a number N and tell them the sum of all the numbers from 1 to N. _Followup_: Write a program that multiplies the numbers from 1 to 100. I.e, calculate: 1 * 2 * 3 * … * 99 * 100 = ?

__Exercise__: Write a program that keeps asking for a password until the user types the secret word.

__Exercise__: Write a program that draws a rectangle with a user specified height and width.

# Part 4: If Statements
An _if statement_ lets us make our program do different things depending on a certain condition. For example, here is a version of our Security Vault program that only lets a single person access the vault:

```python
print("You have reached the security vault.")
name = input("State your name> ")

if name == 'Daniel':
    print(name, "you are cleared to enter.")
else:
    print("Access Denied!")
```

You can check multiple statements with the `and` command:

```python
n = int(input("Enter a number between 1 and 10> "))
if n >= 1 and n <= 10:
    print("Thank you.")
else:
    print("Invalid input.")
```

We can handle more than 2 cases with the `elif` command which is short for _else if_. For example:

```python
age = float(input("Enter your age> "))

if age <= 1:
    print("You are a baby.")
elif age <= 3:
    print("You are a toddler.")
elif age <= 9:
    print("You are a child.")
elif age <= 12:
    print("You are a preteen.")
elif age <= 19:
    print("You are a teenager.")
else:
    print("You are old.")
```

## Guess the Number
We will now make a game called _Guess the Number_. In this game, there is a secret number between 1 and 100 that the player tries to guess. Whenever the player makes a guess, they are told if their guess is too high or too small. They win when they guess the number.

It makes the game more interesting if the secret number is randomly generated every time you play. You can generate random numbers with the `random` module. Here is an example that you can run in the interpreter

```
>>> import random
>>> random.randint(1, 100)
97
>>> random.randint(1, 100)
46
>>> random.randint(1, 100)
76
```

In the first line we tell our program that we would like to use commands related to random numbers. After running this import command, we can use the command `random.randint(1, 100)` to generate a random number between 1 and 100.

Before we make the full game, here is a version that just checks if the player guessed correctly. We also limit the range to be from 1 to 10.

```python
import random

print("Guess the Number")
print("================")

secret = random.randint(1, 10)

print("I am thinking of a number between 1 and 10.")
guess = int(input("Guess a number> "))

while guess != secret:
    print("Incorrect.")
    guess = int(input("Guess a number> "))

print("You got it!")
```

Unfortunately, this version is pretty boring since you may have to guess every number between 1 and 10. This is why in the real game we tell that player if their guess was too high or too low.

__Exercise__: Make the full version of the game! _Bonus_: Limit the number of guesses to 15.

# Part 5: The 21 Game
In this section we are going to make a real game and implement a computer player.

The 21 Game starts with 21 stones on the board. In each turn, a player may remove 1 or 2 stones. The player that removes the last stone wins.

We can display the remaining stones after each turn like this:
```
>>> num_stones = 21
>>> print(num_stones * "# ")
# # # # # # # # # # # # # # # # # # # # #
```

__Project__: Make a program that lets a person play the 21 game against a computer. For this version, let the person make the first move and make the computer play second. Also, program the computer player to  make random moves. _Hint_: Follow the general structure of _Guess the Number_. Just like in that game, you can use a `while` loop that keeps running until the game is over. In each iteration of the loop you first ask the player if they want to remove 1 or 2 stones, print the remaining stones, and then make the computer randomly remove 1 or 2 stones. Make the `while` loop run until there are no more stones. _Bonus_: Make the computer use a better strategy. It turns out that there is a strategy the computer can use that will allow it to win every game!

# Part 6: Functions
In programs that are more than a few lines long it is helpful to define _functions_ which allow you to run many lines in a single command. Here is a program with a function that executes two print statements.

```python
def say_hello():
	print("Hello.")
	print("How are you doing?")

say_hello()
```

When you run this program you will get this output:
```
Hello.
How are you doing?
```

You can also pass a variable to a function like this:
```python
def say_hello(name):
	print("Hello", name)
	print("How are you doing?")

say_hello("Arya")
say_hello("Sansa")
```

When you run this program you will get this output:
```
Hello Arya
How are you doing?
Hello Sansa
How are you doing?
```

Finally, a program can return a value to the place where it was called. Here is an example of a program that uses a function to calculate the area of a square.

```python
def compute_area(height, width):
    """Calculate the area of a rectangle."""
    return height * width


print("AREA CALCULATOR")
print("===============")

h = float(input("height> "))
w = float(input("width> "))

area = compute_area(h, w)
print("area:", area)
```

In this program we first define the function compute_area so that we can use it later to calculate the area of a rectangle. The function has two inputs which are called _arguments_. This first argument is named `height` and the second argument is named `width`. Inside the function, we calculate the area with `height * width`, and use the `return` command to pass the area to the place where the function was called.  Note that we used the triple quotes `"""` to document what the function does. This is an example of a comment and it has no effect on the program.

__Exercise:__ Make a function that calculates the area of a triangle.  
__Exercise:__ Make a function that calculates the absolute value of a number.  
__Exercise:__ Make a function that takes an integer N as input and returns the sum of the numbers from 1 to N. _Bonus_: Return the sum of the even numbers from 2 to N. _Followup_: Write a function that takes two numbers M and N as input and calculates the sum of the numbers from M to N.

# Part 7: Lists
## List Basics
We can use lists to store many objects together. For example:
```
>>> numbers = [10, 11, 12, 13, 14]
>>> print(numbers)
[10, 11, 12, 13, 14]
```

Each element of a list has an _index_. The index of the first element is `0`, the index of the second element is `1` and so on. We can use the index to pick out a particular element from the list. For example, here is how you would get the element in the list `numbers` that has index `0`:

```
>>> numbers[0]
10
```

Here are some more examples:
```
>>> numbers[3]
13
>>> numbers[1]
11
>>> numbers[2]
12
```

We can also store strings in a list. For example:

```
>>> colors = ["red", "orange", "yellow", "green", "blue"]
>>> colors[1]
'orange'
```

It is now easy to assign a player a random color:
```
>>> import random
>>> print("Your color is:", colors[random.randint(0, 4)])
Your color is: yellow
>>> 
```

## The For Loop
We can use a _for loop_ to iterate through the elements of a list. Continuing the example above:

```
>>> for color in colors:
	print(color)
	
red
orange
yellow
green
blue
purple
>>> 
```

You can also use the range command to iterate through a list of numbers:

```
>>> for i in range(5):
	print(i)
0
1
2
3
4
>>> for i in range(5):
	print(i*i)
0
1
4
9
16
```

## Making a Game Board with a Double List
We can store a 2D grid as a list of rows, where each row is itself a list. For example:
```
>>> board = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
>>> print(board)
[[0, 1, 2], [3, 4, 5], [6, 7, 8]]
>>> print(board[0])  # Row 0
[0, 1, 2]
>>> print(board[1])  # Row 1
[3, 4, 5]
>>> print(board[2])  # Row 2
[6, 7, 8]
>>> for row in board:  # Print each row.
	print(row)
[0, 1, 2]
[3, 4, 5]
[6, 7, 8]
>>> print(board[0][0])  # Row 0, column 0
0
>>> print(board[0][1])  # Row 0, column 1
1
>>> print(board[1][0])  # Row 1, column 0
3
>>> print(board[2][1])  # Row 2, column 1
7
```

Note that we used the `#` to write a comment indicating what the code is doing. All text following a `#` has no effect on the program.

Here is an example with a Tic-Tac-Toe board:
```
>>> board = [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]
>>> for row in board:
	print(row)
	
[' ', ' ', ' ']
[' ', ' ', ' ']
[' ', ' ', ' ']
>>> board[1][1] = 'X'  # Add an X to row 1, column 1
>>> for row in board:
	print(row)
	
[' ', ' ', ' ']
[' ', 'X', ' ']
[' ', ' ', ' ']
>>> board[2][0] = 'O'  # Add an O to row 2, column 0
>>> for row in board:  # Print the board
	print(row)

[' ', ' ', ' ']
[' ', 'X', ' ']
['O', ' ', ' ']
```

Let’s make the board look a little nicer:
```
>>> board[0][0] = 'X'
>>> for i in range(3):
	print(board[i][0], '|', board[i][1], '|', board[i][2])
	if i < 2:
		print('---------')
		
X |   |  
---------
  | X |  
---------
O |   |  
```

We can pack this up into a useful function:
```python
def print_board(board):
    for i in range(3):
        print(board[i][0], '|', board[i][1], '|', board[i][2])
        if i < 2:
	    print('---------')
```

It is now easy to print the board whenever it is updated:
```
>>> print_board(board)
X |   |  
---------
  | X |  
---------
O |   |  
>>> board[2][2] = 'O'
>>> print_board(board)
X |   |  
---------
  | X |  
---------
O |   | O
```

# Part 8 - Final Project
Our final project will be to make a Tic-Tac-Toe game where a player can play against a computer opponent. 

It may be helpful to follow the approach we've used in for _Guess the Number_ and _The 21 Game_. In particular, you can use a while loop that keeps running as long as the game is being played. At each iteration of the loop, you can first make the computer select a move, print the board, and then ask the player to make a move. After each move, check if anyone has won the game.

Here is a basic version you can use for reference:
```python
import random

def get_computer_move(board):
    """Choose a square for the computer to put an X. Return the square
       as a list with two elements: [row, column]"""
    row = random.randint(0, 2)
    col = random.randint(0, 2)

    # Keep generating new guesses until we get an available square
    while board[row][col] != ' ':
        row = random.randint(0, 2)
        col = random.randint(0, 2)

    return [row, col]


def check_win(board):
    """Check if anybody has won."""
    # Check if any of the rows have three in a row and are not empty.
    for row in range(3):
        if board[row][0] == board[row][1] == board[row][2] != ' ':
            return True

    # Check if any of the columns have three in a row and are not empty.
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != ' ':
            return True

    # Check the diagonal \
    if board[0][0] == board[1][1] == board[2][2] != ' ':
        return True

    # Check the diagonal /
    if board[2][0] == board[1][1] == board[0][2] != ' ':
        return True

    # If we did not find three in a row
    return False


def print_board(board):
    for i in range(3):
        print(board[i][0], '|', board[i][1], '|', board[i][2])
        if i < 2:
            print('---------')


board = [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]
num_moves = 0

print("Let's play Tic-Tac-Toe! I will play X and you will be O.")
print_board(board)

while True:
    # Make the computer move.
    print("The computer is thinking...")
    square = get_computer_move(board)
    board[square[0]][square[1]] = 'X'
    num_moves = num_moves + 1
    print_board(board)

    # Check for a win or draw.
    if check_win(board):
        print("The computer wins!")
        break
        
    if num_moves == 9:
        print("Draw. Good game!")
        break

    # Get the player's move.
    print("Your turn.")
    row = int(input("Choose a row (0-2)> "))
    col = int(input("Choose a column (0-2)> "))

    if row < 0 or row > 2 or col < 0 or col > 2:
        print("Invalid move! You lose.")
        break

    if board[row][col] != ' ':
        print("That square is already taken! You lose.")
        break
    
    board[row][col] = 'O'
    num_moves = num_moves + 1
    print_board(board)
    
    if check_win(board):
        print("You win!")
        break
```

This relatively short program is a fully functional tic-tac-toe game!

Now you can try to add some simple strategy to the function `make_computer_move`. For example, write some code that checks if the player can win on the next turn and if so, block them. You can also check if the computer has any winning move, and if so, take it.

If you design the computer to always make its first move in the middle and its second move in a corner, then following the simple rules above will guarantee that the computer never loses!

Here is an example of a function that you can use to check if the player can fill a row next turn by placing an O at position [row, col]:
```python
def can_player_fill_row(board, row, col):
    """Check if the player can fill a row by putting
       an O at [row, col].
       For example, if the board looks like this: 
       ['X', ' ', ' ']
       [' ', 'O', 'O']
       [' ', ' ', 'X']

       Then can_player_fill_row(1, 0) should return True since
       The player can fill row 1 by moving to [1, 0].
    """
    # Count the number of Os in the row
    num_O = 0
    for i in range(3):
        if board[row][i] == 'O':
            num_O = num_O + 1

    # The player can win at [row, col] if there are 2 Os in the row and
    # [row, col] is currently empty.
    if num_O == 2 and board[row][col] == ' ':
        return True

    else:
        return False
```

You can make similar functions to check when the player can fill a column or a diagonal. Once you have these, you can put them together to make a single function `can_player_win(board, row, col)` that checks if the player can win next turn by moving to a given position [row, col]. Then, you can add code to the beginning of `make_computer_move` that checks if the player can win in any of the 9 squares. If the player can win in some square [row, col], then make the computer move there! 

Here is a quick outline of some code you can add to the beginning of `make_computer_move`, assuming you’ve implemented `can_player_win`:

```python
# Check if the player can win in any square [row, col]. If they can,
# return this square as the computer's move.
for row in range(3):
    for col in range(3):
        if can_player_win(board, row, col):
	    return [row, col]
```

The code that checks if the computer has a winning move is similar.

Once you have explored what you can do with tic-tac-toe, you can use similar principles to make more complicated board games such as connect four, checkers, chess, go, or what ever your favorite game may be!


