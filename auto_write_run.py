import time
import threading
import multiprocessing
from auto_write_main import auto_write  # 예를 들어 main.py 파일의 run_for_site 함수를 import합니다.

# 각 스레드에서 실행할 함수를 정의합니다.
# 이 함수는 main.py에 정의된 run_for_site 함수를 호출합니다.

def thread_function(site_code):
    auto_write(site_code)  # main.py에 정의된 함수를 호출하고 site_code를 인자로 넘깁니다.

def process_function(site_code):
    auto_write(site_code)

# 메인 함수
if __name__ == "__main__":
    start_time = time.time()
    site_codes = ['2001', '2005', '2030', '2031', '2032']  # 실행할 site_code 리스트
    threads = []
    # processes = []

    # site_codes의 각 사이트 코드에 대해 별도의 스레드를 생성합니다.
    for code in site_codes:
        thread = threading.Thread(target=thread_function, args=(code,))
        threads.append(thread)
        thread.start()

        # process = multiprocessing.Process(target=process_function, args=(code,))
        # processes.append(process)
        # process.start()

    # 모든 스레드가 종료될 때까지 대기합니다.
    for thread in threads:
        thread.join()
    
    # for process in processes:
    #     process.join()

    end_time = time.time()
    print('multi threading time : ', end_time - start_time)