"""
Monkey patch for phone_agent to fix action parsing issues.

Some models return action with XML tags like </answer> that need to be cleaned.
"""


def apply_patches():
    """Apply all patches to phone_agent library."""
    _patch_model_client_parse_response()
    _patch_parse_action()
    _extend_app_packages()


def _patch_model_client_parse_response():
    """
    Patch ModelClient._parse_response to clean XML tags from action.

    Fixes issue where models return:
        do(action="Launch", app="XXX")</answer>
    Instead of:
        do(action="Launch", app="XXX")
    """
    from phone_agent.model.client import ModelClient

    original_parse_response = ModelClient._parse_response

    def patched_parse_response(self, content: str) -> tuple[str, str]:
        thinking, action = original_parse_response(self, content)
        # Clean XML tags from action
        action = (
            action
            .replace("</answer>", "")
            .replace("</think>", "")
            .replace("<answer>", "")
            .replace("<think>", "")
            .strip()
        )
        return thinking, action

    ModelClient._parse_response = patched_parse_response


def _clean_action_string(action: str) -> str:
    """
    Clean XML tags and other common artifacts from action string.

    Handles patterns like:
        do(action="Tap", element=[614,818])</answer>
        <answer>do(action="Tap", element=[614,818])</answer>
    """
    import re

    cleaned = action

    # Remove common XML-like tags
    tags_to_remove = [
        "</answer>", "<answer>",
        "</think>", "<think>",
        "</response>", "<response>",
        "</action>", "<action>",
        "</output>", "<output>",
    ]
    for tag in tags_to_remove:
        cleaned = cleaned.replace(tag, "")

    # Remove any remaining XML-like tags using regex
    cleaned = re.sub(r'</?[a-zA-Z_][a-zA-Z0-9_]*>', '', cleaned)

    return cleaned.strip()


def _patch_parse_action():
    """
    Patch parse_action in handler.py to clean input before parsing.

    This is a more robust fix as it handles the action right before AST parsing,
    regardless of how the action string was produced.
    """
    from phone_agent.actions import handler

    original_parse_action = handler.parse_action

    def patched_parse_action(response: str):
        # Clean the action string before parsing
        cleaned_response = _clean_action_string(response)
        return original_parse_action(cleaned_response)

    handler.parse_action = patched_parse_action


def _extend_app_packages():
    """
    扩展 APP_PACKAGES 字典，添加 phone_agent 库中缺失的应用包名映射。

    phone_agent 库的 Launch 动作依赖 APP_PACKAGES 字典来查找应用包名，
    如果应用不在字典中，launch_app() 会返回 False 导致启动失败。
    """
    from phone_agent.config.apps import APP_PACKAGES

    # 扩展应用包名映射
    # 格式: "应用名": "包名"
    EXTENDED_PACKAGES = {
        # 金融支付
        "支付宝": "com.eg.android.AlipayGphone",
        "Alipay": "com.eg.android.AlipayGphone",
        "alipay": "com.eg.android.AlipayGphone",
        "网商银行": "com.mybank.android.phone",
        "招商银行": "cmb.pb",
        "招行": "cmb.pb",
        "工商银行": "com.icbc",
        "建设银行": "com.chinamworld.main",
        "农业银行": "com.android.bankabc",
        "中国银行": "com.chinamworld.bocmbci",
        "交通银行": "com.bankcomm.Bankcomm",
        "浦发银行": "cn.com.spdb.mobilebank.per",
        "中信银行": "com.ecitic.bank.mobile",
        "光大银行": "com.cebbank.mobile.cemb",
        "平安银行": "com.pingan.paces.ccms",
        "兴业银行": "com.cib.cibmb",
        "民生银行": "com.cmbc.cc.mbank",
        "华夏银行": "com.hxb.mobile",
        "广发银行": "com.cgbchina.xpt",
        "邮储银行": "com.yitong.mbank.psbc",
        "微众银行": "com.webank.wemoney",
        "数字人民币": "cn.gov.pbc.dcep",

        # 证券投资
        "同花顺": "com.hexin.plat.android",
        "东方财富": "com.eastmoney.android.berlin",
        "雪球": "com.xueqiu.android",
        "富途牛牛": "com.futu.futubull",
        "涨乐财富通": "com.htsc.cnoversea",
        "通达信": "com.tdx.androidphone",

        # 办公协作
        "钉钉": "com.alibaba.android.rimet",
        "DingTalk": "com.alibaba.android.rimet",
        "企业微信": "com.tencent.wework",
        "WeChat Work": "com.tencent.wework",
        "飞书": "com.ss.android.lark",
        "Lark": "com.ss.android.lark",
        "腾讯会议": "com.tencent.wemeet.app",
        "Tencent Meeting": "com.tencent.wemeet.app",
        "腾讯文档": "com.tencent.docs",
        "石墨文档": "com.shimo.app",
        "语雀": "com.yuque.app",
        "Notion": "notion.id",
        "印象笔记": "com.yinxiang",
        "Evernote": "com.evernote",
        "有道云笔记": "com.youdao.note",

        # 工具类
        "百度网盘": "com.baidu.netdisk",
        "阿里云盘": "com.alicloud.databox",
        "夸克": "com.quark.browser",
        "UC浏览器": "com.UCMobile",
        "百度": "com.baidu.searchbox",
        "知乎": "com.zhihu.android",
        "豆瓣": "com.douban.frodo",
        "小红书": "com.xingin.xhs",
        "什么值得买": "com.smzdm.client.android",

        # 健康医疗
        "健康码": "com.eg.android.AlipayGphone",  # 通常在支付宝内
        "丁香医生": "com.dxy.healthexpert",
        "好大夫在线": "com.haodf.android",
        "平安健康": "com.phs.hyk.app",

        # 政务服务
        "个人所得税": "cn.gov.tax.its",
        "交管12123": "com.tmri.app.main",
        "铁路12306": "com.MobileTicket",
        "国家政务服务平台": "cn.gov.zwfw",

        # 电商补充
        "唯品会": "com.achievo.vipshop",
        "苏宁易购": "com.suning.mobile.ebuy",
        "当当": "com.dangdang.buy2",
        "网易严选": "com.netease.yanxuan",
        "小米商城": "com.xiaomi.shop",
        "华为商城": "com.vmall.client",

        # 外卖配送
        "饿了么": "me.ele",
        "美团外卖": "com.sankuai.meituan.takeoutnew",
        "盒马": "com.wudaokou.hippo",
        "叮咚买菜": "com.yaya.zone",
        "朴朴超市": "com.pupu.client",
        "每日优鲜": "cn.missfresh.application",

        # 出行补充
        "嘀嗒出行": "com.didapinche.booking",
        "曹操出行": "com.caocaokeji.passengerphone",
        "哈啰": "com.jingyao.easybike",
        "青桔单车": "com.didi.bike",
        "T3出行": "com.t3go.passengerphone",
        "花小猪打车": "com.huaxiaozhu.rider",

        # 社交补充
        "Soul": "cn.soulapp.android",
        "陌陌": "com.immomo.momo",
        "探探": "com.p1.mobile.putong",

        # 视频补充
        "西瓜视频": "com.ss.android.article.video",
        "火山小视频": "com.ss.android.ugc.live",
        "皮皮虾": "com.sup.android.superb",
        "虎牙直播": "com.duowan.kiwi",
        "斗鱼直播": "air.tv.douyu.android",

        # 音乐补充
        "酷狗音乐": "com.kugou.android",
        "酷我音乐": "cn.kuwo.player",
        "喜马拉雅": "com.ximalaya.ting.android",
        "蜻蜓FM": "fm.qingting.qtradio",
        "荔枝FM": "com.yibasan.lizhifm",

        # 游戏平台
        "Steam": "com.valvesoftware.android.steam.community",
        "TapTap": "com.taptap",
        "王者营地": "com.tencent.gamehelper.smoba",
        "掌上英雄联盟": "com.tencent.qt.qtl",
    }

    # 合并到 APP_PACKAGES
    for app_name, package_name in EXTENDED_PACKAGES.items():
        if app_name not in APP_PACKAGES:
            APP_PACKAGES[app_name] = package_name


async def load_custom_app_packages():
    """
    从数据库加载自定义 APP 包名映射并合并到 APP_PACKAGES。
    此函数应在应用启动后调用。
    """
    from phone_agent.config.apps import APP_PACKAGES
    from app.database import async_session
    from app.models.app_package import AppPackage
    from sqlalchemy import select

    async with async_session() as session:
        result = await session.execute(select(AppPackage))
        custom_packages = result.scalars().all()

        for pkg in custom_packages:
            APP_PACKAGES[pkg.app_name] = pkg.package_name


def update_app_package_in_memory(app_name: str, package_name: str):
    """
    实时更新内存中的 APP_PACKAGES 字典（用于创建/更新时）。
    """
    from phone_agent.config.apps import APP_PACKAGES
    APP_PACKAGES[app_name] = package_name


def remove_app_package_from_memory(app_name: str):
    """
    从内存中的 APP_PACKAGES 字典移除指定应用（用于删除时）。
    """
    from phone_agent.config.apps import APP_PACKAGES
    APP_PACKAGES.pop(app_name, None)


def sync_load_custom_app_packages():
    """
    同步方式加载自定义 APP 包名映射（用于任务执行前调用）。
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果事件循环正在运行，创建新任务
            asyncio.create_task(load_custom_app_packages())
        else:
            loop.run_until_complete(load_custom_app_packages())
    except RuntimeError:
        # 没有事件循环，创建新的
        asyncio.run(load_custom_app_packages())

