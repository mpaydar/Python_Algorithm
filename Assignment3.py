# Mohammad Bayat


import random
import time
import pandas as pd
from datetime import datetime
import timeit as tp
import matplotlib.pyplot as plt
from lorem_text import lorem


# Brute-Force-https://www.geeksforgeeks.org/naive-algorithm-for-pattern-searching/
def python_search(arr, pattern):
    return arr.find(pattern)


#https://stackoverflow.com/questions/69350395/brute-force-pattern-algorithm-in-python# brute-force
def brute_search(text, pattern):
    for i in range(len(text)):
        for j in range(len(pattern)):
            if i + j >= len(text):
                break
            if text[i + j] != pattern[j]:
                break
        else:
            return True
    return False




# https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching/
# KMPS
def KMP_search(pat, txt):
    M = len(pat)
    N = len(txt)
    lps = [0] * M
    j = 0
    computeLPSArray(pat, M, lps)

    i = 0
    while i < N:
        if pat[j] == txt[i]:
            i += 1
            j += 1

        if j == M:
            j = lps[j - 1]

        elif i < N and pat[j] != txt[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1


def computeLPSArray(pat, M, lps):
    len = 0

    lps[0]
    i = 1
    while i < M:
        if pat[i] == pat[len]:
            len += 1
            lps[i] = len
            i += 1
        else:
            if len != 0:
                len = lps[len - 1]
            else:
                lps[i] = 0
                i += 1


NO_OF_CHARS = 256
# https://stackoverflow.com/questions/22216948/python-rabin-karp-algorithm-hashing/22218947
def Rabinkarp_search(text, pattern):
    q = 13
    d = 10
    n = len(text)
    m = len(pattern)
    h = pow(d, m - 1) % q
    p = 0
    t = 0
    result = []
    for i in range(m):  # preprocessing
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q
    for s in range(n - m + 1):  # note the +1
        if p == t:  # check character by character
            match = True
            for i in range(m):
                if pattern[i] != text[s + i]:
                    match = False
                    break
            if match:
                result = result + [s]
        if s < n - m:
            t = (t - h * ord(text[s])) % q  # remove letter s
            t = (t * d + ord(text[s + m])) % q  # add letter s+m
            t = (t + q) % q  # make sure that t >= 0
    return result


# https://www.geeksforgeeks.org/boyer-moore-algorithm-for-pattern-searching/
# boyer-moore
NO_OF_CHARS = 256
def badCharHeuristic(string, size):
    badChar = [-1] * NO_OF_CHARS

    for i in range(size):
        badChar[ord(string[i])] = i;

    return badChar


def boyer_moore(txt, pat):
    m = len(pat)
    n = len(txt)
    badChar = badCharHeuristic(pat, m)
    s = 0
    while (s <= n - m):
        j = m - 1

        while j >= 0 and pat[j] == txt[s + j]:
            j -= 1
        if j < 0:
            s += (m - badChar[ord(txt[s + m])] if s + m < n else 1)
        else:

            s += max(1, j - badChar[ord(txt[s + j])])


def random_text(m):
    collect_text = []
    count_character = 0
    str = ""
    text = lorem.sentence()
    for i in text:
        collect_text.append(i)
        count_character += 1
        if count_character == m:
            break
    return str.join(collect_text).upper()




# m=size of text
# n = size of pattern
def random_pattern(m, text):
    index = random.randint(1, len(text)) - m
    random_pattern_generate = text[index:index + m]
    # print(index)
    # print(text[index])
    return random_pattern_generate



def plot_times_bar_graph(sorting_algos, sizes, searches):
    search_num = 0
    plt.xticks([j for j in range(len(sizes))], [str(size) for size in sizes])
    for search in searches:
        search_num += 1
        d = sorting_algos[search.__name__]
        x_axis = [j + 0.09 * search_num for j in range(len(sizes))]
        y_axis = [d[i] for i in sizes]
        plt.bar(x_axis, y_axis, width=.07, alpha=.50, label=search.__name__)
    plt.xticks()
    plt.legend()
    plt.title("Runtime of Searching Algorithms")
    plt.xlabel("Number of elements")
    plt.ylabel("Time for a 100 trials (ms)")
    plt.savefig("search_bargraph.png")
    plt.show()


def main():
    m = 500  # the text size
    n = 20
    trials = 4000
    dict_searches = {}
    searches = [KMP_search, boyer_moore, brute_search, python_search, Rabinkarp_search]
    for search in searches:
        dict_searches[search.__name__] = {}
    sizes = [100, 150, 200, 250, 300, 350,400]
    for size in sizes:
        for search in searches:
            dict_searches[search.__name__][size] = 0
        for trial in range(1, trials):
            arr = random_text(m)  # getting the text
            pattern_t = random_pattern(n, arr)  # pattern text
            for search in searches:
                start_time = time.time()
                search(arr, pattern_t)
                end_time = time.time()
                net_time = end_time - start_time
                dict_searches[search.__name__][size] += net_time

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.options.display.float_format = '{:.5f}'.format
    df = pd.DataFrame.from_dict(dict_searches).T
    print(df)
    # plot_times_line_graph(dict_searches)
    plot_times_bar_graph(dict_searches, sizes, searches)


if __name__ == '__main__':
    main()