#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

'''
Adapted from https://github.com/lernanto/sincomp/blob/1de5c0e64635d304a42f99748ee3248ce1515f7e/scripts/zhongguoyuyan/crawl.py

see https://github.com/lernanto/sincomp/blob/master/src/sincomp/datasets.py#L1159 for an updated version
爬取中国语言资源保护工程采录展示平台数据

@see https://zhongguoyuyan.cn
'''
import sys
import logging
import os
import datetime
import time
import requests
import uuid
import ddddocr
import json
import pandas
import retrying
import subprocess
import rsa
import base64


site = 'https://zhongguoyuyan.cn'
verify_code_url = '{}/api/common/create/verifyCode'.format(site)
login_url = '{}/api/user/login/account'.format(site)
logout_url = '{}/api/user/logout'.format(site)
survey_url = '{}/api/mongo/query/latestSurveyMongo'.format(site)
data_url = '{}/api/mongo/resource/normal'.format(site)
video_info_url = '{}/api/common/getVideoEntry'.format(site)
province_info_url = '{}/api/mongo/search/province'.format(site)

global start_time
global email
global password


# 请求一次资源之后的延迟时间
delay = 10 # seconds between cities
retry_delay = 60 # seconds. to prevent 418
video_delay = 62
relogin = 60 * 60 # seconds. relogging in every hour to prevent 418


def get_verify_code(session):
    '''请求验证码'''

    logging.info('get verify code from {}'.format(verify_code_url))
    rsp = session.get(verify_code_url)
    logging.debug('cookies = {}'.format(session.cookies))

    return rsp.content

def get_token(image, ocr):
    '''从图片识别验证码'''

    # 使用 OCR 识别图片中的验证码
    token = ocr.classification(image)
    logging.info('get token from image, token = {}'.format(token))
    return token

def login(session, email, password, token):
    '''登录到网站'''

    # 发送登录请求
    logging.info('login from {}, email = {}, pass = {}, token = {}'.format(
        login_url,
        email,
        password,
        token
    ))
    rsp = session.post(
        login_url,
        data={'email': email, 'pass': password, 'token': token}
    )
    logging.debug('cookies = {}'.format(session.cookies))

    try:
        ret = rsp.json()
    except requests.exceptions.JSONDecodeError:
        ret = {}
    code = ret.get('code')
    if code == 200:
        # 登录成功
        logging.info('login successful, code = {}'.format(code))
    else:
        # 登录失败
        logging.debug(rsp.content)
        logging.error('login failed, code = {}, {}'.format(
            code,
            ret.get("description")
        ))

    return code

def logout(session):
    '''退出登录'''

    logging.info('logout from {}'.format(logout_url))
    session.get(logout_url)


@retrying.retry(
    retry_on_result=lambda ret: ret.get('status') != 'success',
    stop_max_attempt_number=3,
    wait_exponential_multiplier=retry_delay * 1000
)
def get_survey(session):
    '''获取全部调查点'''

    logging.info('get survey from {}'.format(survey_url))
    rsp = session.post(survey_url)
    survey = rsp.json()

    status = survey.get('status')
    if status == 'success':
        logging.info('get survey sucessful')
    else:
        logging.error('get survey failed, status = {}'.format(status))

    return survey

@retrying.retry(
    retry_on_result=lambda ret: ret.get('code') not in {200, 408, 417},
    stop_max_attempt_number=3,
    wait_exponential_multiplier=retry_delay * 1000
)
def get_site(session, site_id):
    '''获取方言点数据'''

    referer = '{}/point/{}'.format(site, site_id)

    logging.info('get data from {}, ID = {}'.format(data_url, site_id))
    rsp = session.post(data_url, headers={'referer': referer}, data={'id': site_id})
    ret = rsp.json()

    code = ret.get('code')
    if code == 200:
        # 获取数据成功
        logging.debug('get data ID = {} successful, code = {}'.format(site_id, code))
    else:
        logging.debug(ret)
        logging.error('get data ID = {} failed, code = {}, {}'.format(
            site_id,
            code,
            ret.get('description')
        ))

    return ret

def try_login(email, password, retry=3):
    '''尝试登录，如果验证码错误，重试多次'''

    session = requests.Session()
    # 网站要求客户端生成一个 UUID 唯一标识当前会话
    session.cookies.set('uniqueVisitorId', str(uuid.uuid4()))

    # 用于识别验证码
    ocr = ddddocr.DdddOcr()

    for i in range(retry):
        # 尝试登录
        image = get_verify_code(session)
        token = get_token(image, ocr)
        code = login(session, email, password, token)

        # 记录一下验证码图片和识别结果
        dir = 'verify_code'
        os.makedirs(dir, exist_ok=True)
        fname = os.path.join(
            dir,
            '.'.join(('_'.join((
                'zhongguoyuyan',
                uuid.uuid4().hex,
                token,
                '0' if code == 702 else '1'
            )), 'jpg'))
        )
        logging.info('save verify code to {}'.format(fname))
        with open(fname, 'wb') as f:
            f.write(image)

        if code != 702:
            # 不是验证码错误，无论登录成功失败都跳出
            break

    if code == 200:
        logging.info('login successful after {} try'.format(i + 1))
        return session
    else:
        logging.error('login failed after {} try, give up'.format(i + 1))

def parse_survey(survey):
    '''解析调查点 JSON 数据，转换成表格 (dataframe)'''

    objects = {}
    for name in ('dialect', 'minority'):
        obj = pandas.json_normalize(
            survey[name + 'Obj'],
            'cityList',
            'provinceCode'
        ).set_index('_id')
        objects[name] = obj

    return objects

def get_subgrouping(location_data):
    subgrouping = ""
    logging.info('location_data')
    logging.info(location_data)
    if 'area' in location_data and location_data['area']:
        subgrouping += location_data['area']
    if 'slice' in location_data and location_data['slice']:
        subgrouping += location_data['slice']
    if 'slices' in location_data and location_data['slices']:
        subgrouping += location_data['slices']
    if not subgrouping:
        return location_data['dialectInfo']
    return subgrouping

@retrying.retry(
    retry_on_result=lambda ret: ret is None,
    stop_max_attempt_number=3,
    wait_exponential_multiplier=retry_delay * 1000
)
def crawl_video_metadata(session, site_id, video_id):
    rsp = session.get('{}/api/common/getVideoEntry/{}'.format(site, video_id), headers={
        'referer': 'https://zhongguoyuyan.cn/point/{}'.format(site_id),
        'host': 'zhongguoyuyan.cn',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'
    })
    logging.debug('cookies = {}'.format(session.cookies))

    try:
        ret = rsp.json()
        code = ret.get('code')
    except requests.exceptions.JSONDecodeError:
        # probably a 500
        code = rsp.status_code
        ret = {}
    if code == 200:
        # 成功
        logging.info('video metadata getVideoEntry successful, code = {}'.format(code))
        return ret.get('data', {})
    else:
        # 失败
        logging.debug(rsp.content)

        logging.error('video metadata getVideoEntry failed, code = {}, {}'.format(
            code,
            ret.get("description")
        ))
        return None

def crawl_video(session, site_id, video_id, site_name, prefix):
    # skip if already downloaded
    if os.path.exists(f"{prefix}/{site_name}/{video_id}.mp4"):
        logging.debug(f"{prefix}/{site_name}/{video_id}.mp4 exists")
        return

    try:
        metadata = crawl_video_metadata(session, site_id, video_id)
    except retrying.RetryError as e:
        metadata = None

    if metadata is None:
        return False
    url = metadata.get('url', None)
    if url is None:
        return False

    os.makedirs(f"{prefix}/{site_name}", exist_ok=True)

    time.sleep(delay)

    wget_call = subprocess.run([
        "wget",
        "--header",
        "Referer: https://zhongguoyuyan.cn/",
        "-O",
        f"{prefix}/{site_name}/{video_id}.mp4",
        url
    ])
    if wget_call.returncode != 0:
        logging.error(f"Failed to download {site_name} {video_id}.mp4")
        logging.error(wget_call.stderr)
        return False
    return True

def parse_site(site_data):
    # subgrouping info
    location_data = site_data['mapLocation']['location']
    site_name = f"{location_data['province']}_{location_data['city']}_{location_data['country']}"
    subgrouping = get_subgrouping(location_data)

    resources = [resource for resource in site_data['resourceList'] if resource['type'] == '语法']
    if len(resources) < 1:
        logging.error(site + ' does not have any sentence (语法) recordings')
        return None
    sentence_recordings = resources[0]

    sentences = {}
    for recording in sentence_recordings['items']:
        # missing video
        # if recording['video'] == '':
        #     continue
        speaker_id = recording['oid']
        sentence_id = recording['iid']
        transcript, ipa, translation = '', '', ''
        # there could be multiple ways of saying the same sentence
        for record in recording['records']:
            transcript += record['memo']
            ipa += record['phonetic']
            translation += record['sentence']
        sentences[speaker_id + sentence_id] = {
            'transcript': transcript,
            'translation': translation,
            'ipa': ipa,
            'english': recording['en_name']
        }
    return {
        'site': site_name,
        'subgrouping': subgrouping,
        'sentences': sentences
    }

def crawl_survey(session, prefix='.', update=False):
    '''爬取调查点列表并保存到文件'''

    # 如果调查点列表文件已存在，且指定不更新，则使用现有文件中的数据
    survey_file = os.path.join(prefix, 'survey.json')
    if os.path.exists(survey_file) and not update:
        logging.info(
            'survey data file {} exits. load without crawling'.format(survey_file)
        )
        with open(survey_file, encoding='utf-8') as f:
            survey = json.load(f)

    else:
        # 调查点列表文件不存在，或指定更新文件，从网站获取最新调查点列表
        try:
            survey = get_survey(session)
        except:
            return

        logging.info('save survey data to {}'.format(survey_file))
        with open(survey_file, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(survey, f, ensure_ascii=False, indent=4)

    return parse_survey(survey)

def crawl_site(session, site_id, insert_time, record, new_crawl, prefix='.'):
    '''爬取一个调查点的数据并保存到文件'''

    try:
        code = record.loc[site_id, 'code']
        modify_time = record.loc[site_id, 'modify_time']
    except KeyError:
        code = None
        modify_time = datetime.datetime.fromtimestamp(0)

    data_file = os.path.join(prefix, '{}.json'.format(site_id))

    # TODO: add back code == 200 and (insert_time is None or modify_time > insert_time)
    if os.path.exists(data_file):
        # 上次爬取时间晚于调查点数据加入时间，数据已是最新，不再重复爬取
        logging.debug(
            'data ID = {} is already present. do not get site data, {}'.format(
                site_id,
                data_file
            )
        )
        with open(data_file, 'r') as f:
            site_data = json.load(f)

        if not os.path.exists(os.path.join(prefix, site_data['site'])):
            logging.debug('but the videos do not exist')

            if not site_data['sentences']:
                # no scraping was done
                return False

            # NOTE: the video scraping method below is not reliable, so we remove it. see README.md
            # for video_id in site_data['sentences']:
            #     global start_time, email, password
            #     if time.time() - start_time > relogin:
            #         start_time = time.time()
            #         session = try_login(email, rsa_encrypt(password))
            #         time.sleep(retry_delay)
            #     crawl_video(session, site_id, video_id, site_data['site'], prefix)
            #     time.sleep(video_delay)

            # scraping was done
            return True
        
        # no scraping was done
        return False

    elif code is not None and code != 200:
        # 由于资源的原因爬取不成功，暂时不再尝试
        logging.debug('resource unavailable, code = {}, {}'.format(
            code,
            record.loc[site_id, 'description']
        ))
        # no scraping was done
        return False

    else:
        logging.info('get data ID = {}'.format(site_id))
        try:
            ret = get_site(session, site_id)
        except retrying.RetryError as e:
            ret = e.last_attempt

        code = ret.get('code')
        if code in {200, 417}:
            # 记录爬取结果
            new_crawl.append({
                'id': site_id,
                'modify_time': datetime.datetime.now(),
                'code': code,
                'description': ret.get('description')
            })

        if code == 200:
            site_data = ret.get('data')
            site_data = parse_site(site_data)
            if not site_data:
                # scraping was done
                return True

            logging.info('save data to {}'.format(data_file))
            with open(data_file, 'w', encoding='utf-8', newline='\n') as f:
                json.dump(site_data, f, ensure_ascii=False, indent=4)

            # for video_id in site_data['sentences']:
                # if time.time() - start_time > relogin:
                #     start_time = time.time()
                #     session = try_login(email, password)
                #     time.sleep(retry_delay)
            #     crawl_video(session, site_id, video_id, site_data['site'], prefix)
            #     time.sleep(video_delay)
            return ret
        else:
            # scraping was done
            return True


def load_record(fname):
    '''加载之前的爬取记录'''

    if os.path.exists(fname):
        logging.info('load crawling records from {}'.format(fname))
        record = pandas.read_csv(fname, index_col='id', encoding='utf-8')
    else:
        logging.info(
            'crawling record file {} does not exist. assume empty'.format(fname)
        )
        record = pandas.DataFrame()
        record.index.rename('id', inplace=True)

    return record

def save_record(fname, record):
    '''保存爬取记录'''

    logging.info('save crawling record to {}'.format(fname))
    record.to_csv(fname, encoding='utf-8')


def crawl(
    email,
    password,
    prefix='.',
    provinces=[],
    update_survey=False,
    max_success=2000,
    max_fail=10
):
    '''爬取调查点数据'''

    success_count = 0
    fail_count = 0

    logging.info('try creating directory {}'.format(prefix))
    os.makedirs(prefix, exist_ok=True)

    # 登录网站
    session = try_login(email, rsa_encrypt(password))
    if session:
        # 爬取调查点列表
        survey = crawl_survey(session, prefix)

        if survey:
            # 加载爬取记录
            record_file = os.path.join(prefix, 'record.csv')
            record = load_record(record_file)
            new_crawl = []

            stop = False
            global start_time
            start_time = time.time()
            for site_id, row in survey['dialect'].sort_index().iterrows():
                # skip sites not in provinces of interest
                if len(provinces) > 0 and row['province'] not in provinces:
                    continue

                # relogin every hour or so to prevent 418
                if time.time() - start_time > relogin:
                    start_time = time.time()
                    session = try_login(email, rsa_encrypt(password))
                    time.sleep(retry_delay)

                ret = crawl_site(
                    session,
                    site_id,
                    row['insertDate'] if type(row['insertDate']) is datetime.datetime else None,
                    record,
                    new_crawl,
                    prefix,
                )

                if ret:
                    # if scraping was done, then delay
                    # important to prevent 418 (rate limiting) / 408 (account suspension)
                    time.sleep(delay)

                # if ret is not None:
                #     if ret:
                #         success_count += 1
                #         if success_count % 100 == 0:
                #             logging.info('crawled {} data'.format(success_count))

                #         if success_count >= max_success:
                #             # 达到设置的最大爬取数量
                #             logging.info(
                #                 'reached maximum crawl number = {}, have a rest'.format(
                #                     max_success
                #                 )
                #             )
                #             stop = True
                #             break

                #     else:
                #         fail_count += 1
                #         if fail_count >= max_fail:
                #             # 多次爬取失败，不再尝试
                #             logging.error(
                #                 'reached maximum failure number = {}, abort'.format(
                #                     max_fail
                #                 )
                #             )
                #             stop = True
                #             break

            if not stop:
                logging.info(
                    'all data crawled, nothing else todo, success_count = {}'.format(
                        success_count
                    )
                )

            # 把新的爬取记录写回文件
            if new_crawl:
                # ignore_index=True resets the indices
                record = pandas.concat([record, pandas.DataFrame(new_crawl).set_index('id')], ignore_index=True)
                save_record(record_file, record)

        else:
            # 获取调查点列表失败
            logging.error('cannot get survey. exit')

        # 退出登录
        logout(session)

    logging.info('totally crawl {} data'.format(success_count))


def rsa_encrypt(raw_pass):
    base64_public_key = 'MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBAKUBBGEwEB6Bpm1W/kBNQ8EHEK7D+aitDOztwtHnepflduBBsTI7gmohhsP6uhRKlVaIkndnwW0fSvDxUueI29sCAwEAAQ=='
    der_bytes = base64.b64decode(base64_public_key)
    public_key = rsa.PublicKey.load_pkcs1_openssl_der(der_bytes)

    hashed_pass = rsa.encrypt(raw_pass.encode(), public_key)
    return base64.b64encode(hashed_pass).decode()


def main():
    logging.basicConfig(
        level=logging.DEBUG,  # Set the minimum log level you want to see
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]  # Ensures output goes to stdout (the notebook)
    )

    global email, password
    email, password = sys.argv[2], rsa_encrypt(sys.argv[3])
    prefix = sys.argv[4] # directory to save videos to

    provinces = [
        '福建', # Min, Hakka
        '海南', # Min
        '浙江', # Wu, Min, Hakka, Mandarin
        '上海', # Wu
        '广东', # Yue, Hakka, Pinghua
        '台湾', # Min, Hakka
        '澳门', # Yue
        '香港', # Yue
        '广西', # Yue, Mandarin, Pinghua
        '江苏', # Wu
        '安徽', # Gan, Hui, Wu, Mandarin
        '江西', # Gan
        '湖北', # Gan, Mandarin
        '湖南', # Xiang, Mandarin, Gan, Hakka, Tuhua, Xianghua
        # Mandarin
        '北京',
        '重庆',
        '甘肃',
        '贵州',
        '河北',
        '河南',
        '黑龙江',
        '吉林',
        '江苏',
        '辽宁',
        '内蒙古',
        '宁夏',
        '青海',
        '山东',
        '山西',
        '陕西',
        '四川',
        '天津',
        '西藏',
        '新疆',
        '云南'
    ]
    # excluded 濒危方言

    crawl(email, password, prefix, provinces)


if __name__ == '__main__':
    main()
