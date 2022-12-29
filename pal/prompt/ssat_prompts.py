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
Q: At Nicholas's Computer World, computers usually sold for $ 1,500 are now being sold for $ 1,200. What fraction of the original price is the new price? (A) $\frac {1}{10}$ (B) $\frac {1}{5}$ (C) $\frac {3}{4}$ (D) $\frac {4}{5}$ (E) $\frac {7}{10}$

# solution in Python:


def solution():
    """At Nicholas's Computer World, computers usually sold for $ 1,500 are now being sold for $ 1,200. What fraction of the original price is the new price? (A) $\frac {1}{10}$ (B) $\frac {1}{5}$ (C) $\frac {3}{4}$ (D) $\frac {4}{5}$ (E) $\frac {7}{10}$"""
    import math
    import sympy
    options = [1/10, 1/5, 3/4, 4/5, 7/10]
    previous_computer_price = 1500
    current_computer_price = 1200
    price_fraction = current_computer_price / previous_computer_price
    correct_option = None
    for i, option in enumerate(options):
        if math.fabs(option - price_fraction) < 1e-4:
            correct_option = chr(ord('A') + i)
            break
    result = correct_option
    return result





Q: Andre had a birthday party and spent $12.98 on balloons, $47.23 on party favors, $22.97 on a cake, $14.77 on ice cream, and $15.00 on invitations. How much did Andre spend on the party? (A) $87.25 (B) $112.95 (C) $125.20 (D) $127.30 (E) $131.50

# solution in Python:


def solution():
    """Andre had a birthday party and spent $12.98 on balloons, $47.23 on party favors, $22.97 on a cake, $14.77 on ice cream, and $15.00 on invitations. How much did Andre spend on the party? (A) $87.25 (B) $112.95 (C) $125.20 (D) $127.30 (E) $131.50"""
    import math
    import sympy
    options = [87.25, 112.95, 125.20, 127.30, 131.50]
    spent_on_balloons = 12.98
    spent_on_party_favors = 47.23
    spoent_on_cake = 22.97
    spent_on_ice_cream = 14.77
    spent_on_invitations = 15.00
    total_spent_on_party = spent_on_balloons + spent_on_party_favors + spoent_on_cake + spent_on_ice_cream + spent_on_invitations
    correct_option = None
    for i, option in enumerate(options):
        if math.fabs(option - total_spent_on_party) < 1e-4:
            correct_option = chr(ord('A') + i)
            break
    result = correct_option
    return result





Q: If $3x-y=6$, then what does $y+4$ equal? (A) $3x-6$ (B) $3x-2$ (C) $3x+10$ (D) $4x-2$ (E) It cannot be determined from the information given.

# solution in Python:


def solution():
    """If $3x-y=6$, then what does $y+4$ equal? (A) $3x-6$ (B) $3x-2$ (C) $3x+10$ (D) $4x-2$ (E) It cannot be determined from the information given."""
    import math
    import sympy
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    options = [3* x - 6, 3*x - 2, 3*x + 10, 4*x - 2, None]
    eq1 = sympy.Eq(3 * x - y, 6)
    sol = sympy.solve([eq1], [y + 4])
    solved_result = sol[y + 4]
    correct_option = None
    for i, option in enumerate(options):
        if option == solved_result:
            correct_option = chr(ord('A') + i)
            break
    if correct_option is None and None in options:
        correct_option = "E"
    result = correct_option
    return result





Q: If Christophe ran 3 miles in half an hour, his average speed was (A) 1.5 miles per hour (B) 3 miles per hour (C) 4.5 miles per hour (D) 6 miles per hour (E) 12 miles per hour

# solution in Python:


def solution():
    """If Christophe ran 3 miles in half an hour, his average speed was (A) 1.5 miles per hour (B) 3 miles per hour (C) 4.5 miles per hour (D) 6 miles per hour (E) 12 miles per hour"""
    import math
    import sympy
    options = [1.5, 3, 4.5, 6, 12]
    distance = 3
    time = 0.5
    average_speed = distance / time
    correct_option = None
    for i, option in enumerate(options):
        if math.fabs(option - average_speed) < 1e-4:
            correct_option = chr(ord('A') + i)
            break
    result = correct_option
    return result





Q: What is 36 expressed as the product of prime factors? (A) (2)(3) (B) (3)(12) (C) (2)(2)(3)(3) (D) (4)(9) (E) (6)(6)

# solution in Python:


def solution():
    """What is 36 expressed as the product of prime factors? (A) (2)(3) (B) (3)(12) (C) (2)(2)(3)(3) (D) (4)(9) (E) (6)(6)"""
    import math
    import sympy
    options = [[2, 3], [3, 12], [2, 2, 3, 3], [4, 9], [6, 6]]
    target_num_to_be_factored = 36
    def isPrime(n):
        # Corner case
        if (n <= 1):
            return False
        # Check from 2 to sqrt(n)
        for i in range(2, int(math.sqrt(n))+1):
            if (n % i == 0):
                return False
        return True
    correct_option = None
    for i, option in enumerate(options):
        # check if all numbers are prime numbers
        all_is_prime = True
        product = 1
        for num in option:
            if not isPrime(num):
                all_is_prime = False
                break
            else:
                product = product * num
        if all_is_prime and math.fabs(product - target_num_to_be_factored) < 1e-4:
            correct_option = chr(ord('A') + i)
            break
    result = correct_option
    return result





Q: A glass of cold water is placed on a table at room temperature. If it starts at a temperature of $ 1 ^{\circ }C$ and increases to $21^{\circ }C$ in the course of 4 hours, what is the average rise in temperature per hour? A. $3^{\circ }$ B. $4^{\circ }$ C. $5^{\circ }$ D. $6^{\circ }$ E. $7^{\circ }$

# solution in Python:


def solution():
    """A glass of cold water is placed on a table at room temperature. If it starts at a temperature of $ 1 ^{\circ }C$ and increases to $21^{\circ }C$ in the course of 4 hours, what is the average rise in temperature per hour? A. $3^{\circ }$ B. $4^{\circ }$ C. $5^{\circ }$ D. $6^{\circ }$ E. $7^{\circ }$"""
    import math
    import sympy
    options = [3, 4, 5, 6, 7]
    target_num_to_be_factored = 36
    start_temperature = 1
    temperature_after_increase = 21
    duration = 4
    temperature_difference = temperature_after_increase - start_temperature
    average_temperature_rise_per_hour = temperature_difference / duration
    correct_option = None
    for i, option in enumerate(options):
        if math.fabs(option - average_temperature_rise_per_hour) < 1e-4:
            correct_option = chr(ord('A') + i)
            break
    result = correct_option
    return result





Q: $-6(3-4\times 3)=$ (A) -66 (B) -54 (C) -12 (D) 18 (E) 54

# solution in Python:


def solution():
    """$-6(3-4\times 3)=$ (A) -66 (B) -54 (C) -12 (D) 18 (E) 54"""
    import math
    import sympy
    options = [-66, -54, -12, 18, 54]
    calculated_result = -6 * ( 3 - 4 * 3)
    correct_option = None
    for i, option in enumerate(options):
        if math.fabs(option - calculated_result) < 1e-4:
            correct_option = chr(ord('A') + i)
            break
    result = correct_option
    return result





Q: If 40 percent of a movie ticket costs $5.00, what is 20 percent of the cost of two tickets? (A) $2.50 (B) $5.00 (C) $6.00 (D) $7.50 (E) $10.00

# solution in Python:


def solution():
    """If 40 percent of a movie ticket costs $5.00, what is 20 percent of the cost of two tickets? (A) $2.50 (B) $5.00 (C) $6.00 (D) $7.50 (E) $10.00"""
    import math
    import sympy
    options = [2.50, 5.00, 6.00, 7.50, 10.00]
    movie_ticket_partial_price = 5
    movie_ticket_partial_rate = 40 / 100
    movie_ticket_price = movie_ticket_partial_price / movie_ticket_partial_rate
    two_movie_ticket_price = 2 * movie_ticket_price
    two_movie_ticket_price_partial_rate = 20 / 100
    two_movie_ticket_price_partial_price = two_movie_ticket_price_partial_rate * two_movie_ticket_price
    correct_option = None
    for i, option in enumerate(options):
        if math.fabs(option - two_movie_ticket_price_partial_price) < 1e-4:
            correct_option = chr(ord('A') + i)
            break
    result = correct_option
    return result





'''