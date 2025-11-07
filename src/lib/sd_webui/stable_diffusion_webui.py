#!/bin/env python
import sys
import json
import aiohttp
import asyncio
import logging
import requests
from src.lib.etc import log
from typing import Optional

# POST /sdapi/v1/txt2img
txt2img_base = {
    "prompt": "",
    "seed": -1,
    "subseed_strength": 0,
    "sampler_name": "Euler a",
    "batch_size": 1,
    "n_iter": 1,
    "steps": 40,
    "cfg_scale": 7,
    "width": 640,
    "height": 960,
    "restore_faces": False,
    "tiling": False,
    "do_not_save_samples": False,
    "do_not_save_grid": False,
    "negative_prompt": "",
    "s_churn": 0,
    "s_tmax": 0,
    "s_tmin": 0,
    "s_noise": 1,
    "override_settings": {},
    "override_settings_restore_afterwards": True,
    "send_images": True,
    "save_images": False,
    "alwayson_scripts": {}
}

txt2img_hr = {
    "enable_hr": False,
    "denoising_strength": 0,
    "firstphase_width": 0,
    "firstphase_height": 0,
    "hr_scale": 2,
    "hr_upscaler": "string",
    "hr_second_pass_steps": 0,
    "hr_resize_x": 0,
    "hr_resize_y": 0,
}


# For x/y/z grid

# Available Axis options
# See xyz_grid.py line 201
# Keep only lines that are of type AxisOption()
XYZGridAvailableScripts = [
    "Nothing",
    "Checkpoint name",
    "VAE",
    "Dict name",
    "Prompt S/R",
    "Styles",
    "Sampler",
    "Sampler",
    "Seed",
    "Steps",
    "CFG Scale",
    "Var. seed",
    "Var. strength",
    "Clip skip",
    "Denoising",
    "Hires steps",
    "Image CFG Scale",
    "Prompt order",
    "Sampler Sigma Churn",
    "Sampler Sigma min",
    "Sampler Sigma max",
    "Sampler Sigma noise",
    "Sampler Eta",
    "Sampler Solver Order",
    "Face restore",
    "Token merging ratio",
    "Token merging ratio high-res",
    "SecondPass Sampler",
    "SecondPass Denoising Strength",
    "SecondPass Steps",
    "SecondPass CFG Scale",
    "SecondPass Guidance Rescale",
    "SecondPass Refiner Start"
]


def xy_script_args(XAxisType="Nothing", XAxisValues="", XAxisDropdown = "",
                   YAxisType="Nothing", YAxisValues="", YAxisDropdown = "",
                   ZAxisType="Nothing", ZAxisValues="", ZAxisDropdown = "",
                   drawLegend="True", includeSeparateImages="False",
                   IncludeSubGrids = "True", NoFixedSeed="False",
                   MarginSize="0", NoGrid="False"):
    return [
        XYZGridAvailableScripts.index(XAxisType), XAxisValues, XAxisDropdown,
        XYZGridAvailableScripts.index(YAxisType), YAxisValues, YAxisDropdown,
        XYZGridAvailableScripts.index(ZAxisType), ZAxisValues, ZAxisDropdown,
        drawLegend, includeSeparateImages,
        IncludeSubGrids, NoFixedSeed, MarginSize, NoGrid
    ]



"""
helper methods that creates HTTP session with managed connection pool
provides async HTTP get/post methods and several helper methods
"""


sd_url = "http://127.0.0.1:7860" # automatic1111 api url root
use_session = True
timeout = aiohttp.ClientTimeout(total = None, sock_connect = 10, sock_read = None) # default value is 5 minutes, we need longer for training
sess: Optional[aiohttp.ClientSession] = None
quiet = False


async def session():
    global sess # pylint: disable=global-statement
    time = aiohttp.ClientTimeout(total = None, sock_connect = 10, sock_read = None) # default value is 5 minutes, we need longer for training
    sess = aiohttp.ClientSession(timeout = time, base_url = sd_url)
    log.debug({ 'sdapi': 'session created', 'endpoint': sd_url })
    """
    sess = await aiohttp.ClientSession(timeout = timeout).__aenter__()
    try:
        async with sess.get(url = f'{sd_url}/') as req:
            log.debug({ 'sdapi': 'session created', 'endpoint': sd_url })
    except Exception as e:
        log.error({ 'sdapi': e })
        await asyncio.sleep(0)
        await sess.__aexit__(None, None, None)
        sess = None
    return sess
    """
    return sess


async def result(req):
    if req.status != 200:
        if not quiet:
            log.error({ 'request error': req.status, 'reason': req.reason, 'url': req.url })
        if not use_session and sess is not None:
            await sess.close()
        return { 'error': req.status, 'reason': req.reason, 'url': req.url }
    else:
        json = await req.json()
        if type(json) is list:
            res = json
        elif json is None:
            res = {}
        else:
            res = json
        log.debug({ 'request': req.status, 'url': req.url, 'reason': req.reason })
        return res


def resultsync(req: requests.Response):
    if req.status_code != 200:
        if not quiet:
            log.error({ 'request error': req.status_code, 'reason': req.reason, 'url': req.url })
        return { 'error': req.status_code, 'reason': req.reason, 'url': req.url }
    else:
        json = req.json()
        if type(json) is list:
            res = json
        elif json is None:
            res = {}
        else:
            res = json
        log.debug({ 'request': req.status_code, 'url': req.url, 'reason': req.reason })
        return res


async def get(endpoint: str, json: Optional[dict] = None):
    global sess # pylint: disable=global-statement
    sess = sess if sess is not None else await session()
    try:
        async with sess.get(url = endpoint, json = json) as req:
            res = await result(req)
            return res
    except Exception as err:
        log.error({ 'session': err })
        return {}


def getsync(endpoint: str, json: Optional[dict] = None):
    try:
        req = requests.get(f'{sd_url}{endpoint}', json = json) # pylint: disable=missing-timeout
        res = resultsync(req)
        return res
    except Exception as err:
        log.error({ 'session': err })
        return {}



async def post(endpoint: str, json: Optional[dict] = None):
    global sess # pylint: disable=global-statement
    # sess = sess if sess is not None else await session()
    if sess and not sess.closed:
        await sess.close()
    sess = await session()
    try:
        async with sess.post(url = endpoint, json = json) as req:
            res = await result(req)
            return res
    except Exception as err:
        log.error({ 'session': err })
        return {}


def postsync(endpoint: str, json: Optional[dict] = None):
    req = requests.post(f'{sd_url}{endpoint}', json = json) # pylint: disable=missing-timeout
    res = resultsync(req)
    return res


async def interrupt():
    res = await get('/sdapi/v1/progress?skip_current_image=true')
    if isinstance(res, dict) and 'state' in res and isinstance(res['state'], dict) and res['state'].get('job_count', 0) > 0:
        log.debug({ 'interrupt': res['state'] })
        res = await post('/sdapi/v1/interrupt')
        await asyncio.sleep(1)
        return res
    else:
        log.debug({ 'interrupt': 'idle' })
        return { 'interrupt': 'idle' }


def interruptsync():
    res = getsync('/sdapi/v1/progress?skip_current_image=true')
    if isinstance(res, dict) and 'state' in res and isinstance(res['state'], dict) and res['state'].get('job_count', 0) > 0:
        log.debug({ 'interrupt': res['state'] })
        res = postsync('/sdapi/v1/interrupt')
        return res
    else:
        log.debug({ 'interrupt': 'idle' })
        return { 'interrupt': 'idle' }


async def progress():
    res = await get('/sdapi/v1/progress?skip_current_image=true')
    log.debug({ 'progress': res })
    return res


def progresssync():
    res = getsync('/sdapi/v1/progress?skip_current_image=true')
    log.debug({ 'progress': res })
    return res


def options():
    options = getsync('/sdapi/v1/options')
    flags = getsync('/sdapi/v1/cmd-flags')
    return { 'options': options, 'flags': flags }


def shutdown():
    try:
        postsync('/sdapi/v1/shutdown')
    except Exception as e:
        log.info({ 'shutdown': e })

async def close():
    if sess is not None:
        await asyncio.sleep(0)
        await sess.__aexit__(None, None, None)
        log.debug({ 'sdapi': 'session closed', 'endpoint': sd_url })

if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    if 'interrupt' in sys.argv:
        asyncio.run(interrupt())
    if 'progress' in sys.argv:
        asyncio.run(progress())
    if 'options' in sys.argv:
        opt = options()
        log.debug({ 'options' })
        print(json.dumps(opt['options'], indent = 2))
        log.debug({ 'cmd-flags' })
        print(json.dumps(opt['flags'], indent = 2))
    if 'shutdown' in sys.argv:
        shutdown()
    asyncio.run(close())