# using uvicorn as testserver
# start testserver with CUDA_VISIBLE_DEVICES=3 uvicorn server:app --reload --port 8001
# CUDA_VISIBLE_DEVICES=3 python server.py
# CUDA_VISIBLE_DEVICES=3 python -m my_project.server.server_tests # lsof -i :8001
import torch
import aiohttp
import asyncio
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    CYELLOW2 = '\33[93m'

def print_newTest(test:str):
    print(bcolors.CYELLOW2+30*"-")
    print(test)
    print(30*"-"+bcolors.ENDC)
    
def print_endTest():
    print(bcolors.CYELLOW2+30*"-"+bcolors.ENDC)

req1 = "Paris is the capital city of"
req2 = "Berlin is the capital city of Germany"
params1 = {
    "reqs": req1,
    "gen_lens": [32]
}
params2 = {
    "reqs": "Dummy Request 2", 
    "gen_lens": [12],
}
params3 = {
    "reqs": ["Dummy Req 3", "Dummy Req 3"],
    "gen_lens": [4, 18]
}
params4 = {
    "reqs": [req1,req1],
    "gen_lens": [12, 12]
}
params5 = {
    "reqs": [req1,req1],
    "gen_lens": [128, 128]
}

async def make_req(session, params):
    res = await session.get("/queue", params=params)
    print(await res.text())

async def main():
    addr = "http://127.0.0.1:8002/"
    async with aiohttp.ClientSession(addr) as session:
        print_newTest("queue request tests")
        with torch.no_grad():
            # await make_req(session, {})
            # await make_req(session, {"reqs:": ""})
            # await make_req(session, params1)
            # await make_req(session, params2)
            # await make_req(session, params3)
            # await make_req(session, params4)
            await make_req(session, params5)
        print_endTest()

if __name__ == "__main__":
    asyncio.run(main())