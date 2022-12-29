# Copyright 2022 PAL Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


MATH_PROMPT = '''
Q: If $|6a - 2 | = 3$ , what is a possible value of a ? (A) 3 (B) -3 (C) 29 (D) $-\\frac {1} {3}$ (E) $\\frac {5}{6}$

# parsing result:
(Abs[6*a - 2], =, 3)
(a, =, ?)
{3, -3, 29, -(1/3), 5/6}



Q: Divya and two of her friends planned to spend $120 each on gas for their road trip. Then, another friend decided to join them. If all friends re-divided the cost of gas equally among them, how much did each friend spend? (A) $80 (B) $90 (C) $100 (D) $110 (E) $120

# parsing result:
(spent_per_person, =, 120)
(people_before, =, 3)
(people_after, =, 4)
(spent_per_person * people_before / people_after, =, ?)
{80, 90, 100, 110, 120}



Q: Which of the following is not even? (A) 330 (B) 436 (C) 752 (D) 861 (E) 974

# parsing result:
(Mod[?, 2], =, 1)
{330, 436, 752, 861, 974}



Q: Brad bought an MP3 player on sale at a 20% discount from its regular price of $118. If there is an 8% sales tax that is calculated on the sale price, how much did Brad pay? (A) $23.60 (B) $86.85 (C) $94.40 (D) $101.95 (E) $127.44

# parsing result:
(regular_price, =, 118)
(discount_rate, =, 20/100)
(sales_tax_rate, =, 8/100)
(discounted_price, =, regular_price * (1-discount_rate))
(discounted_price * (1 + sales_tax_rate), =, ?)
{23.60, 86.85, 94.40, 101.95, 127.44}



Q: If $(x-y)+2=6$ and y is less than 3 , which of the following CANNOT be the value of x ? (A) -3 (B) 0 (C) $1\\frac {1}{2}$ (D) 4 (E) 8

# parsing result:
((x-y)+2, =, 6)
(y, <, 3)
(x, !=, ?)
{-3, 0, 1.5, 4, 8}



Q: The blue team has 12 players and the red team has 20 players. How many players need to move from the red team to the blue team in order for the teams to have the same number of players? (A) 10 (B) 8 (C) 6 (D) 4 (E) 3

# parsing result:
(blue_team_num, =, 12)
(red_team_num, =, 20)
(blue_team_num + move_num, =, red_team_num - move_num)
(move_num, =, ?)
{10, 8, 6, 4, 3}



Q: $71\\frac{1}{5}\\% =$ (A) 712 (B) 71.2 (C) 7.12 (D) 0.712 (E) 0.0712

# parsing result:
((71 + 1/5) / 100, =, ?)
{712, 71.2, 7.12, 0.712, 0.0712}



Q: Which of the following can be expressed as $(J+2)\\times 3$ where J is a whole number? (A) 40 (B) 52 (C) 65 (D) 74 (E) 81

# parsing result:
(j, is, Integers)
((j + 2) * 3, =, ?)
{40, 52, 65, 74, 81}



Q: If 40 percent of a movie ticket costs $5.00, what is 20 percent of the cost of two tickets? (A) $2.50 (B) $5.00 (C) $6.00 (D) $7.50 (E) $10.00

# parsing result:
(40/100 * ticket_price, =, 5)
(20/100 * 2 * ticket_price, =, ?)
{2.50, 5.00, 6.00, 7.50, 10.00}



Q: The sum of five consecutive positive integers is 55. What is the square of the greatest of these integers? (A) 5 (B) 9 (C) 13 (D) 81 (E) 169

# parsing result:
(x + (x+1) + (x+2) + (x+3) + (x+4), =, 55)
((x+4)^2, =, ?)
{5, 9, 13, 81, 169}



Q: One school bus can transport 48 students. If 218 students need to be transported, how many buses are needed? (A) 4 (B) 4 $\frac {13}{24}$ (C) 5 (D) 5 $\frac {5}{8}$ (E) 6 $\frac {1}{3}$


# parsing result:
(students_per_bus, =, 48)
(students_needed, =, 218)
(Ceiling[students_needed / students_per_bus], =, ?)
{4, 4 + 13/24, 5, 5 + 5/8, 6 + 1/3}



Q: Which of the following is NOT a multiple of 4? (A) 20 (B)30 (C)36 (D)44 (E)96

# parsing result:
(Mod[?, 4], !=, 0)
{20, 30, 36, 44, 96}



Q: {question}

# parsing result:
'''.strip()