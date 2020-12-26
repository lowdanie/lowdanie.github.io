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

{% include video.html src="/assets/sound_waves/sine_wave.mp4" %}
____

[^UNSW]: Speech and Helium Speech [http://newt.phys.unsw.edu.au/jw/speechmodel.html](http://newt.phys.unsw.edu.au/jw/speechmodel.html)

[^HCP]: John R. Rumble, ed., "CRC Handbook of Chemistry and Physics, 101st Edition" (Internet Version 2020), _CRC Press/Taylor & Francis_, Boca Raton, FL. [http://hbcponline.com/faces/documents/14_19/14_19_0003.xhtml](http://hbcponline.com/faces/documents/14_19/14_19_0003.xhtml)
