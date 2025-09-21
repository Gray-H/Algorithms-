import math
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from enum import Enum

app = FastAPI(
    title="Array Search API",
    description="Search for a target value in an array using selected algorithm with execution timing.",
    version="1.1.0"
)

class AlgorithmEnum(str, Enum):
    linear = "linear"
    binary = "binary"
    jump = "jump"
    interpolation = "interpolation"

class SearchRequestSingle(BaseModel):
    array: List[int]
    target: int
    algorithm: AlgorithmEnum

class SearchRequestAll(BaseModel):
    array: List[int]
    target: int

class SearchResult(BaseModel):
    algorithm: str
    index: int
    time_seconds: float
    found: bool

class SearchResponseSingle(BaseModel):
    request: SearchRequestSingle
    result: SearchResult

class SearchResponseAll(BaseModel):
    request: SearchRequestAll
    results: List[SearchResult]

class ArraySearcher:
    def __init__(self, arr, target):
        self.arr = arr
        self.target = target

    def linear_search(self):
        start_time = time.perf_counter()
        for index, value in enumerate(self.arr):
            if value == self.target:
                duration = time.perf_counter() - start_time
                return index, duration
        duration = time.perf_counter() - start_time
        return -1, duration

    def binary_search(self):
        start_time = time.perf_counter()
        if len(self.arr) == 0:
            return -1, time.perf_counter() - start_time

        sorted_arr = sorted(self.arr)
        left, right = 0, len(sorted_arr) - 1

        while left <= right:
            mid = (left + right) // 2
            if sorted_arr[mid] == self.target:
                try:
                    original_index = self.arr.index(self.target)
                    duration = time.perf_counter() - start_time
                    return original_index, duration
                except ValueError:
                    return -1, time.perf_counter() - start_time
            elif sorted_arr[mid] < self.target:
                left = mid + 1
            else:
                right = mid - 1

        return -1, time.perf_counter() - start_time

    def jump_search(self):
        start_time = time.perf_counter()
        n = len(self.arr)
        if n == 0:
            return -1, time.perf_counter() - start_time

        sorted_arr = sorted(self.arr)
        step = int(math.sqrt(n))
        prev = 0

        while prev < n and sorted_arr[min(step, n) - 1] < self.target:
            prev = step
            step += int(math.sqrt(n))
            if prev >= n:
                return -1, time.perf_counter() - start_time

        for i in range(prev, min(step, n)):
            if sorted_arr[i] == self.target:
                try:
                    original_index = self.arr.index(self.target)
                    duration = time.perf_counter() - start_time
                    return original_index, duration
                except ValueError:
                    return -1, time.perf_counter() - start_time

        return -1, time.perf_counter() - start_time

    def interpolation_search(self):
        start_time = time.perf_counter()
        n = len(self.arr)
        if n == 0:
            return -1, time.perf_counter() - start_time

        sorted_arr = sorted(self.arr)
        low, high = 0, n - 1

        while low <= high and sorted_arr[low] <= self.target <= sorted_arr[high]:
            if low == high:
                if sorted_arr[low] == self.target:
                    try:
                        original_index = self.arr.index(self.target)
                        duration = time.perf_counter() - start_time
                        return original_index, duration
                    except ValueError:
                        return -1, time.perf_counter() - start_time
                return -1, time.perf_counter() - start_time

            if sorted_arr[high] == sorted_arr[low]:
                break

            pos = low + int(
                ((float(high - low) / (sorted_arr[high] - sorted_arr[low])) * (self.target - sorted_arr[low]))
            )

            if pos < low or pos > high:
                break

            if sorted_arr[pos] == self.target:
                try:
                    original_index = self.arr.index(self.target)
                    duration = time.perf_counter() - start_time
                    return original_index, duration
                except ValueError:
                    return -1, time.perf_counter() - start_time
            elif sorted_arr[pos] < self.target:
                low = pos + 1
            else:
                high = pos - 1

        return -1, time.perf_counter() - start_time

    def search_one(self, algo_name: str):
        if algo_name == "linear":
            index, exec_time = self.linear_search()
        elif algo_name == "binary":
            index, exec_time = self.binary_search()
        elif algo_name == "jump":
            index, exec_time = self.jump_search()
        elif algo_name == "interpolation":
            index, exec_time = self.interpolation_search()
        else:
            raise ValueError("Unsupported algorithm")

        return SearchResult(
            algorithm=algo_name.capitalize() + " Search",
            index=index,
            time_seconds=exec_time,
            found=index != -1
        )

    def search_all(self):
        results = []
        for algo in ["linear", "binary", "jump", "interpolation"]:
            result = self.search_one(algo)
            results.append(result)
        return results

@app.post("/search", response_model=SearchResponseAll, summary="Search using all algorithms")
async def search_all_endpoint(request: SearchRequestAll):
    searcher = ArraySearcher(request.array, request.target)
    results = searcher.search_all()
    return SearchResponseAll(request=request, results=results)

@app.post("/search/one", response_model=SearchResponseSingle, summary="Search using one selected algorithm")
async def search_one_endpoint(request: SearchRequestSingle):
    searcher = ArraySearcher(request.array, request.target)
    try:
        result = searcher.search_one(request.algorithm)
        return SearchResponseSingle(request=request, result=result)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid algorithm selected.")

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to Array Search API. Go to /docs to try it out."}