from email.mime.multipart import MIMEMultipart
import logging
import os
from pathlib import Path
from random import random
import smtplib
import ssl
import time
import json
import subprocess

import requests
from typing_extensions import Annotated
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from langchain_experimental.utilities import PythonREPL
from pydantic import BaseModel
from langchain_core.tools import tool
from sqlalchemy import inspect
from sqlalchemy import text
from typing import List, Dict, Optional
from functools import wraps

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# 调试
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def send_qq_email(
    to_email: str,
    subject: str,
    body: str,
    *,
    attachment_paths: Optional[List[str]] = None,
    is_html: bool = True,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    use_ssl: bool = True  #  SSL/STARTTLS
) -> str:
    """
    QQ emial send tool
    Args:
        -to_email: str,
        -subject: str,
        -body: str,
        *,
        -attachment_paths: Optional[List[str]] = None,
        -is_html: bool = True,
        -cc: Optional[str] = None,
        -bcc: Optional[str] = None,
        -use_ssl: bool = True  
    """

    QQ_EMAIL = ''
    QQ_APP_PASSWORD = ''  

    msg = MIMEMultipart("alternative")
    msg["From"] = QQ_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    if cc: msg["Cc"] = cc
    if bcc: msg["Bcc"] = bcc

    # 正文
    msg.attach(MIMEText(body, "plain", "utf-8"))
    if is_html: msg.attach(MIMEText(body, "html", "utf-8"))

    # 附件
    if attachment_paths:
        for file_path in attachment_paths:
            file_path = Path(file_path).expanduser().resolve()
            if not file_path.exists():
                logger.warning(f"附件不存在: {file_path}")
                continue
            with open(file_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={file_path.name}")
            msg.attach(part)

    # 发送逻辑（增强）
    server = None
    try:
        if use_ssl:
            # 首选 SSL 465
            context = ssl.create_default_context()  # 新增：现代 TLS
            server = smtplib.SMTP_SSL("smtp.qq.com", 465, context=context, timeout=30)
        else:
            # 备用 STARTTLS 587
            server = smtplib.SMTP("smtp.qq.com", 587, timeout=30)
            context = ssl.create_default_context()
            server.starttls(context=context)

        logger.info(f"连接 QQ SMTP {'(SSL)' if use_ssl else '(STARTTLS)'}...")
        server.login(QQ_EMAIL, QQ_APP_PASSWORD)
        logger.info("登录成功")

        recipients = [to_email]
        if cc: recipients.extend([x.strip() for x in cc.split(",")])
        if bcc: recipients.extend([x.strip() for x in bcc.split(",")])
        server.sendmail(QQ_EMAIL, recipients, msg.as_string())
        logger.info("邮件发送成功")

        return json.dumps({
            "status": "success",
            "message": f"邮件发送成功 → {to_email}",
            "recipients": len(recipients)
        }, ensure_ascii=False)

    except Exception as e:
        error_msg = f"邮件发送失败: {str(e)}"
        logger.error(error_msg)
        return json.dumps({
            "status": "error",
            "message": error_msg,
            "tip": "检查授权码、网络，或试用_ssl=False"
        }, ensure_ascii=False)

    finally:
        # 新增：优雅关闭（关键修复！）
        if server:
            try:
                server.quit()
                logger.info("SMTP 连接已关闭")
            except:
                logger.warning("关闭连接时出错（忽略）")

print(send_qq_email("1595205151@qq.com", "Test", "Hello", use_ssl=True))