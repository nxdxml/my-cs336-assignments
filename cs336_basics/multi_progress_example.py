from multiprocessing import Process, Queue

def worker(task_data, output_queue):
    """
    子进程执行的函数，处理任务并将结果放入队列。
    """
    # 模拟任务处理
    result = task_data ** 2  # 假设任务是求平方
    output_queue.put(result)  # 结果放入队列返回给主进程

def main():
    # 1. 创建进程间通信队列
    q = Queue()

    # 2. 创建任务列表
    tasks = [1, 2, 3, 4, 5]

    # 3. 创建并启动多个进程
    processes = []
    for task in tasks:
        p = Process(target=worker, args=(task, q))
        p.start()
        processes.append(p)

    # 4. 从队列收集所有子进程的结果
    results = [q.get() for _ in processes]

    # 5. 等待所有子进程结束
    for p in processes:
        p.join()

    print("Results:", results)

if __name__ == "__main__":
    main()