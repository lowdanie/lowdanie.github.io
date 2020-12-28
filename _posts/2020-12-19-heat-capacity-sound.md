---
layout: post
title: "Heat Capacity and the Speed of Sound"
date: 2020-12-19
mathjax: true
---
It is well known that sound waves travel through different gasses at different speeds. Perhaps the most popular demonstration of this phenomenon are the modulated voices of people who have inhaled Helium [^UNSW].

The goal of this post is to understand how the molecular composition of a gas determines the speed of sound waves passing through it. To make sure we are grounded in reality, we would specifically like to explain this table of the speed of sound through various gasses at standard atmospheric pressure (1 atm) and the indicated reference temperatures from the Handbook of Chemistry and Physics [^HCP]:

| Name | Formula | Molar Mass (g/mol) | Temperature (C) | Speed (m/s) |
| :--- | :------ | :----------------- | :-------------- | :---------- |
| Carbon Dioxide | $CO_2$ | 44.01 | 0 | 258 |
| Ammonia | $NH_3$ | 17.031 | 0 | 414 |
| Neon | $Ne$ | 20.179 | 0 | 433 |
| Helium | $He$ | 4 | 0 | 973 | 
| Ethane | $C_2H_6$ | 30.07 | 27 | 312 |
| Argon | $Ar$ | 39.948 | 27 | 323 |
| Oxygen | $O_2$ | 31.999 | 27 | 330 |
| Dry Air | | 28.96 | 25 | 346 |
| Nitrogen | $N_2$ | 28.01 | 27 | 353 |
| Methane | $CH_4$ | 16.04 | 27 | 450 | 
| Hydrogen | $H_2$ | 2.01 | 27 | 1320 |

After staring at this table a bit a few trends emerge. First of all, the speed of sound seems to decrease as the mass of the gas increases. But clearly mass is not the only relevant factor. For example, Argon is heavier than Ethane but sound travels more quickly through Argon. The same is true for Neon and Ammonia.

How can we make sense of these outliers? Note that in both cases the lighter molecule was nevertheless up of a larger number of atoms. For example, a molecule of Ethane has 8 atoms whereas a molecule of Argon has only one. So perhaps the number of atoms is what determines the speed of sound, where a greater number of atoms implies a lower speed?

This second hypothesis fails as well. For example, Methane also has more atoms than Argon but the speed of sound in Methane is greater.

We can tentatively conclude that a robust model for the speed of sound in a gas probably has to account for both its molar mass _and_ the number of atoms per molecule.

It turns out that that the speed of sound in an ideal gas depends on three factors: temperature, molar mass and heat capacity. As we will explain later in more detail, heat capacity is essentially a proxy for the number of atoms. The general idea is that molecules can absorb energy into the vibrations of their constituent atoms relative to one another. So molecules with more atoms have more opportunities to absorb heat.

Our goal for the rest of this post will be to understand why these three factors determine the speed of sound, and use this intuition to develop an equation which can reproduce the speeds listed in the table.

In the next section we will study a simple model consisting of a wave passing through a series of balls connected by springs. As we will see, the speed of such a wave depends on the mass of the balls and the stiffness of the springs.

We will then replace these springs by pneumatic tubes filled with gas. Using the ideal gas law and the first law of thermodynamics we'll see that the "stiffness" of such a spring is determined by the temperature and heat capacity of the gas.

Finally, we'll put everything together to write down a formula for the speed of sound in a gas in terms of its mass, temperature and heat capacity and verify that it is consistent with the table.

# The Ball and Spring Model
Sounds waves are an example of a _compression wave_. In this section we will study a simpler model of compression waves and extend our results to sound waves later.

Our model consists of a series of balls connected by springs as in the diagram below:

The balls all have the same mass which we denote by $m$. Similarly, all of the springs have the same stiffness $k$ and length $h$.

What happens if we squeeze the springs on one of the ends? Here is a simulation comparing two systems with 100 balls that differ only in the spring stiffness. Each line represents a ball and the springs have been omitted for clarity. Also, a few of the "balls" have been colored red to make it easier to track their motion but they are physically identical to the rest of the balls.

{% include video.html src="/assets/sound_waves/spring_pair.mp4"
   width="900" height="600" %}

The simulation starts with the ten leftmost springs compressed. These springs push back, compressing the springs to their right. Then those springs push back against their neighbors and eventually the springs on the right side become compressed. We call this propagation of compression a _compression wave_.

Note that when the wave in the bottom model with $k=10$ reaches the other side (i.e, when the right end is compressed), the top model with $k=1$ has only gone about $1/3$ of the way. In other words, the wave in the $k=10$ model travels about $3$ times as fast.

The goal for the remainder of this section will be to derive a formula for computing the speed of a compression wave in the ball and springs model in terms of the mass $m$, spring stiffness $k$ and spring length $h$. If the formula is any good then hopefully it will predict the speed difference we observed in the simulation.

Let's start by analyzing how far each ball deviates from equilibrium at time $t$. In our model, the equilibrium position for ball $i$ is $x = i \times h$. We define $u(x, t)$ to be equal to the amount that the ball with equilibrium position $x$ has moved by time $t$. For example, $u(4*h, 10)$ records the amount that ball $4$ has moved from its equilibrium at time $t=10$.

To make things more concrete, here is a graph of $u(x, t)$ for the $k=10$ model in the previous simulation. (To make the numbers nicer to look at, in this simulation we are measuring $x$ in terms of multiples of $h$ and $t$ is the frame index).

{% include video.html src="/assets/sound_waves/spring_u.mp4"
   width="900" height="600" %}

As you can see, at $t=0$, $u(10, 0) = -9$ which corresponds to the fact that the $10$th ball has been pulled to the left by $9\times h$ meters. The balls nearby have also been shifted to a lesser degree. But $u(x, 0) = 0$ for $x \geq 20$ which means that all the balls following ball $20$ start in their equilibrium position. After $30$ frames things are different. Now $u(10, 30)$ is close to zero whereas $u(90, 30)$ is around $5$ indicating that now the rightmost springs are compressed.

It may be helpful to focus on one of the lines highlighted in red and observe how the dot above it on the graph moves up and down as the line vibrates left and right.

To understand how compression waves propagate we will analyze how $u(x, t)$ evolves over time. Let's start by focusing on the ball with equilibrium position $x$:

How much force is exerted on this ball at time $t$? First lets consider "spring 1" in the diagram. The right end of the spring has been displaced by $u(x, t)$ and the left end has been displaced by $u(x-h, t)$. Together this means that spring 1 has been stretched by $u(x, t) - u(x-h, t)$. Therefore by Hooke's Law the force exerted by spring 1 on the ball is:
\\[
F_1 = k \cdot (u(x, t) - u(x-h, t))   
\\]

Similarly, the force exerted by spring 2 is:
\\[
F_2 = k \cdot (u(x + h, t) - u(x, t))   
\\]

Together we find that the force on the ball is:
\\[
F = F_1 + F_2 = k \cdot (u(x + h, t) - 2u(x, t) + u(x - h, t))
\\]

Multiplying and dividing by $h^2$ we get:
\\[
F = k h^2 \cdot \frac{(u(x + h, t) - 2u(x, t) + u(x - h, t))}{h^2}
\\]

But as $h\rightarrow 0$, the fraction on the right converges to the second derivative of $u(x, t)$ with respect to $x$ so:
\begin{equation}\label{eq:force}
    F = kh^2 \frac{\partial^2 u}{\partial x^2}
\end{equation}

By Newton's second law, the force on the ball is related to the acceleration by:
\begin{equation}\label{eq:accel}
    F = m \cdot \frac{\partial^2 u}{\partial t^2}
\end{equation}

Finally, let's define a constant $c$ by:
\\[
    c = \sqrt{\frac{kh^2}{m}}
\\]

Combining equations \ref{eq:force} and \ref{eq:accel} we get the famous _wave equation_:
\\[
    \frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}
\\]

We'll now show that the speed of the wave is actually equal to $c$! To see why this is so, let's try to find some functions $u(x, t)$ that satisfy the wave equation. As a warm up, here is one super simple solution:
\\[
    u(x, t) = x - c\cdot t
\\]

To verify that this is a valid solution, we calculate some derivatives and find that indeed:
\\[ 
    \frac{\partial^2}{\partial t^2}(x - c\cdot t) = 
    c^2 = 
     c^2 \frac{\partial^2}{\partial x^2}(x - c\cdot t)
\\]

With a bit more [analysis](https://en.wikipedia.org/wiki/Wave_equation#General_solution) one can show that _all_ solutions to the wave equation are of the form:
\\[
u(x, t) = f(x + ct) + g(x - ct)
\\]

where $f$ and $g$ are arbitrary single variable functions. But notice that $g(x - ct)$ represents the function $g(x)$ moving to the right with speed $c$ and $f(x + ct)$ is the function $f(x)$ moving to the left with speed $t$. In conclusion, solutions $u(x, t)$ to the wave equation propagate with speed $c$.

To summarize, we have found that compression waves propagate through the ball and spring model with a speed of:
\begin{equation}\label{eq:c}
    c = \sqrt{\frac{kh^2}{m}}
\end{equation}

How well does this explain our simulations at the beginning of the section? Recall that we observed that the wave in the model with $k=10$ travelled around three times faster than the wave in the $k=1$ model. On the other hand our formula predicts that if $m$ and $h$ are fixed, the speed of the wave with be proportional to $\sqrt{k}$. In our case, $\sqrt{10}$ is indeed close to $3$!

# Making a Spring out of Gas
We now relate the ball and spring model to sound by replacing the springs with tubes of gas. Consider a tube which is filled with gas and sealed at the end by a piston. As part of this model we will also assume that the the tube is thermally insulated from the environment. 

[[ DIAGRAM ]]

In the equilibrium state, the pressure in the tube is equal to the atmospheric pressure so the piston is stationary. But if the piston is pulled out, the pressure in the tube decreases and so the piston is pushed back in. Similarly, if the piston is pressed in the pressure inside increases and the piston is pushed back out. The net result is that if the piston is displaced, it oscillates around the equilibrium point like a spring:

[[ SIMULATED DIAGRAM ]]

The goal of this section will be to complete this analogy by computing the _stiffness_ $k$ of a pneumatic tube in terms of properties of the gas inside.

By definition, stiffness measures the degree to which the piston resists deformation. To be concrete, suppose we slightly displace the piston by $dx$ meters. Then the force pushing the piston back will be equal to

\begin{equation}\label{eq:piston-hooke}
dF = -k \cdot dx
\end{equation}

So to compute $k$ we must determine how the change in force force $dF$ depends on the change in position $dx$.

[[ DIAGRAM ]]

Let $A$ denote the area of the piston, $p_a$ denote the atmospheric pressure and $p$ denote the pressure in the tube. Then the force on the piston is equal to $F = A \cdot (p - p_a)$. Therefore:
\begin{equation}\label{eq:piston-dF}
dF = A \cdot dp
\end{equation}

Furthermore, let $V$ denote the volume of the tube and let $V_0$ denote the volume of the piston at equilibrium. Then $V = V_0 + A\cdot dx$ which means that:
\begin{equation}\label{eq:piston-dx}
dx = \frac{dV}{A}
\end{equation}

Putting together equations \ref{eq:piston-hooke}, \ref{eq:piston-dF} and \ref{eq:piston-dx} we find that $A \cdot dp = -k \cdot \frac{dV}{A}$ or in other words:
\begin{equation}\label{eq:piston-k}
k = -A^2 \frac{dp}{dV}
\end{equation}

Thus, to compute the stiffness $k$ me must compute the derivative of the pressure with respect to volume at the equilibrium. Before we continue, note that $-\frac{dp}{dV}$ is indeed a very reasonable measure of stiffness. If $-\frac{dp}{dV}$ is big then compressing the volume of the tube by $dV$ will cause a large increase in pressure inside the tube meaning that then pressing on the tube will be very difficult.

Recall that the interior of the tube is thermally insulated by the environment. This type of process is called _adiabatic_. A fundamental result in thermodynamics states that an ideal gas going through an adiabatic process satisfies the following equation [^WIKI]:
\begin{equation}\label{eq:adiabatic}
pV^{\gamma} = \mathrm{constant}
\end{equation}

where $\gamma$ is the _adiabatic index_ of the gas [^WIKI2]. We will discuss the adiabatic index in more detail later, but for now we'll simply note that it is roughly inversely proportional to the gas's heat capacity.

Differentiating equation \ref{eq:adiabatic} with respect to $V$ around the equilibrium values of $V = V_0$ and $p = p_0$ gives us:
\\[
V_0^\gamma \frac{dp}{dV} + p_0\gamma V_0^{\gamma - 1} = 0
\\]

which means that
\begin{equation}\label{eq:dpdV}
\frac{dp}{dV} = -\frac{p_0 \gamma}{V_0}
\end{equation}

Plugging this into equation \ref{eq:piston-k} gives us our formula for the stiffness of a pneumatic tube:
\begin{equation}\label{eq:piston-k-final}
k = \gamma A^2 \frac{p_0}{V_0}
\end{equation}

# Calculating the Speed of Sound
We will now merge the ball and spring model with the pneumatic tubes to compute the speed of sound in a gas. Specifically, we want to compute the speed of a sound wave propagating through a long cylinder of gas with cross section $A$. 

Sound waves are examples of compression waves. For instance, suppose someone claps their hand on the left side of the cylinder. This will displace push on the gas at that end and cause it to become compressed. This compressed gas will push back and end up compressing the gas next to it and so on. The end result is a compression wave which is very similar to the ones we studied earlier with the ball and spring model.

To push the analogy further, lets subdivide the cylinder of gas into many small tubes each with length $h$:

[[ DIAGRAM ]]

As a crude approximation, we will assume that the mass $m$ in each tube is concentrated on its left wall. In the previous section we saw that a tube of gas reacts to compression like a spring.  We can thus model the cylinder of gas as a series of balls (the walls) connected by springs (the tubes of gas).

What is the stiffness coefficient of the springs? Let $p_0$ denote the equilibrium pressure of the gas and $V_0 = hA$ equilibrium volume. By equation \ref{eq:piston-k-final} the stiffness is:

\\[
k = \gamma A^2 \frac{p_0}{V_0}
\\]

By equation \ref{eq:c} this means that the speed of compression waves propagating through the gas is:
\\[
    c = \sqrt{\gamma A^2 \frac{p_0}{m V_0} h^2}
\\]

Using the relationship $V_0 = hA$ we can simplify this to:
\begin{equation}\label{eq:sound-c}
    c = \sqrt{\gamma \frac{p_0 V_0}{m} }
\end{equation}

where as before, $m$ is the mass of the gas in each tube.

According to the ideal gas law [^WIKI3]:
\\[
    pV = nRT
\\]

where $n$ is the number of molecules in a tube (in moles), $R = 8.314\, \mathrm{JK}^{-1}\mathrm{mol}^{-1}$ is the ideal gas constant and $T$ is the temperature in Kelvin. Plugging this into equation \ref{eq:sound-c} leaves us with:

\begin{equation}\label{eq:sound-c-2}
    c = \sqrt{\gamma \frac{nRT}{m} }
\end{equation}

Finally, let $M$ be the mass of one mole of gas molecules measured in kilograms. By definition this means that $M$ is equal to the mass of a tube divided by the number of molecules in the tube in moles: $M = \frac{m}{n}$. We can therefore simplify equation \ref{eq:sound-c-2} to:

\begin{equation}\label{eq:sound-c-3}
    c = \sqrt{\gamma \frac{RT}{M} }
\end{equation}

Now that we have a precise formula, we can test it on the table of speeds that we puzzled over in the introduction! Here is another version of the table where we have added a column for the adiabatic index which we looked up in HCP [^HCP] and a column containing the speed of sound as predicted by equation \ref{eq:sound-c-3}.

| Name | M (kg/mol) | $\gamma$ | T (K) |Speed (m/s) | $\sqrt{\gamma \frac{RT}{M} }$ |
| :--- | :--------- | :------- | :---- | :--------- | :-----------------------------|
| Carbon Dioxide  | 0.044 |  1.289 | 273.15 | 258 | 257 | 
| Ammonia         | 0.017 |  1.310 | 273.15 | 414 | 418 | 
| Neon            | 0.020 |  1.666 | 273.15 | 433 | 434 | 
| Helium          | 0.004 |  1.666 | 273.15 | 973 | 972 | 
| Ethane          | 0.030 |  1.188 | 300.15 | 312 | 314 | 
| Argon           | 0.039 |  1.666 | 300.15 | 323 | 326 | 
| Oxygen          | 0.031 |  1.394 | 300.15 | 330 | 335 | 
| Nitrogen        | 0.028 |  1.400 | 300.15 | 353 | 353 | 
| Methane         | 0.016 |  1.304 | 300.15 | 450 | 450 | 
| Hydrogen        | 0.002 |  1.406 | 300.15 | 1320 | 1324 |

The agreement between the experimentally observed speeds and our formula is quite remarkable! This reinforces our theory that the speed of sound in an ideal gas depends only on the temperature, mass of the molecules and the adiabatic index of the gas.

It makes sense that wave would have a harder time traveling through a gas with heavier molecules. But what does the adiabatic index have to do with it? In the next section we will discuss the index in more detail and develop some intuition for why a higher adiabatic index implies a faster speed of sound.

# Heat Capacity and the Adiabatic Index

In  the previous section we showed that in order to calculate the speed of sound in a gas one must know its adiabatic index. We will now provide an interpretation for this index and explain what it has to do with the speed of sound. The emphasis of this section will on  intuitive rather than formal explanations. 

The adiabatic index of an ideal gas can be expressed in terms of the gas's heat capacity [^WIKI2]:
\begin{equation}\label{eq:gamma-cv}
\gamma = 1 + \frac{R}{c_V}
\end{equation}

As before, $R$ is the ideal gas constant. The interesting term here is $c_V$ which denotes the heat capacity of one mole of gas at a constant volume - also known by the fancier name of the _isochoric specific heat_.

In concrete terms, suppose you have one mole of gas molecules in a sealed container. By definition, raising the temperature of the gas by $1$ degree Kelvin requires $c_V$ Joules of energy.

For example Helium ($He$) has a specific heat of $c_V = 12.486$ so raising the temperature of one mole of Helium by $1\,K$ requires $12.486$ Joules of energy. On the other hand, the specific heat of Ethane ($C_2H_6$) is $c_V = 44.186$ so far more energy is required to raise the temperature of Ethane by the same amount.

What is the reason for this difference? It ultimately comes down to the number of ways in which a molecule of the gas can move. In a monatomic (i.e, each molecule has a single atom) gas like Helium each molecule can move in the $x$, $y$ and $z$ directions so we say that it has $3$ _degrees of freedom_. In contrast, an each Oxygen molecule ($O_2$) has two atoms. So in addition to the molecule being able to move linearly in $3$ directions, the two atoms can rotate relative to one another in two ways (e.g $\phi$ and $\psi$ in spherical coordinates) so Oxygen has a total of $5$ degrees of freedom. Finally, each Ethane molecule has $8$ atoms which means that it has even more degrees of freedom.

Now, the temperature of a gas is roughly proportional to the average energy of motion in each degree of freedom. So in order to raise the temperature we must increase the energy _in each degree of freedom_ of the gas. Since Ethane has more degrees of freedom than Helium it takes more energy to raise its temperature. This explains why Ethane has a higher specific heat.

We can use this degrees of freedom interpretation to explain the relationship between specific heat and the speed of sound. Recall that in our discussion of the ball and spring model we found that the speed of compression waves is proportional to the _stiffness_ of the springs. In the case of sound waves the springs can be modeled as tubes filled with gas. By equations \ref{eq:piston-k-final} and  \ref{eq:gamma-cv} the stiffness of such a tube is inversely proportional to the specific heat.

So we can rephrase our question as: Why are gasses with a higher specific heat easier to compress?

Consider two pneumatic tubes, one filled with Helium ($He$) and one with Ethane ($C_2H_6$). 

[[ DIAGRAM ]]

Suppose we now press down on the pistons by the same amount, thus increasing the energy of the gas. In the case of Helium, all of this energy will be spent on increasing the kinetic energy of the gas molecules since, as we mentioned earlier, these are the only degrees of freedom. On the other hand, the Ethane gas will use part of the energy to increase the kinetic energy, and use the rest of it to make the molecules spin around:

[[ DIAGRAM ]]

The net result is that Helium molecules will end up moving faster, thus raising the pressure more quickly compared to the Ethane gas. The difference came down to the fact that Helium molecules have fewer degrees of freedom, and hence a smaller heat capacity, than Ethane molecules. This provides some intuition for equation \ref{eq:dpdV} which played a key role in our calculation of the stiffness of a tube of gas:
\\[
\frac{dp}{dV} = -\gamma \frac{p_0}{V_0} = -(1 + \frac{R}{c_V})\frac{p_0}{V_0}
\\]

For a formal derivation I suggest looking at the wikipedia page [^WIKI].

In summary, we've seen that the speed of sound in a gas depends on its temperature, mass and specific heat. If the gas molecules have a larger mass then they are harder to move which makes sound travel more slowly. If the gas has a higher specific heat then it reacts less forcefully to compression which also causes sound to travel more slowly. We formalized this relationship with an explicit equation (\ref{eq:sound-c-3}) and found that it agrees quite nicely with experimental evidence!



____

[^UNSW]: Speech and Helium Speech [http://newt.phys.unsw.edu.au/jw/speechmodel.html](http://newt.phys.unsw.edu.au/jw/speechmodel.html)

[^HCP]: John R. Rumble, ed., "CRC Handbook of Chemistry and Physics, 101st Edition" (Internet Version 2020), _CRC Press/Taylor & Francis_, Boca Raton, FL. [http://hbcponline.com/](http://hbcponline.com/)

[^WIKI]: [https://en.wikipedia.org/wiki/Adiabatic_process#Ideal_gas_(reversible_process)](https://en.wikipedia.org/wiki/Adiabatic_process#Ideal_gas_(reversible_process))

[^WIKI2]: [https://en.wikipedia.org/wiki/Heat_capacity_ratio](https://en.wikipedia.org/wiki/Heat_capacity_ratio)

[^WIKI3]: [https://en.wikipedia.org/wiki/Ideal_gas_law](https://en.wikipedia.org/wiki/Ideal_gas_law)

[^WIKI4]: [https://en.wikipedia.org/wiki/Speed_of_sound#Speed_of_sound_in_ideal_gases_and_air](https://en.wikipedia.org/wiki/Speed_of_sound#Speed_of_sound_in_ideal_gases_and_air)