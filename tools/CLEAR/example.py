from multiprocessing import Pool


def f(x):
    return x * x


def main():
    pool = Pool(processes=3)  # set the processes max number 3
    result = pool.map(f, range(10))
    pool.close()
    pool.join()
    print(result)
    print('end')


if __name__ == "__main__":
    main()
