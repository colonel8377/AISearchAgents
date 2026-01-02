from typing import Optional, List


class ParserConfig:
    """Configuration for file parsers."""

    def __init__(self,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 supported_extensions: Optional[List[str]] = None,
                 enable_cloud_fallback: bool = True,
                 cloud_timeout: int = 30,
                 local_timeout: int = 60,
                 privacy_keywords: Optional[List[str]] = None):
        self.max_file_size = max_file_size
        self.supported_extensions = supported_extensions or []
        self.enable_cloud_fallback = enable_cloud_fallback
        self.cloud_timeout = cloud_timeout
        self.local_timeout = local_timeout
        # Enhanced privacy keywords with comprehensive coverage
        self.privacy_keywords = privacy_keywords or self._get_default_privacy_keywords()

    @staticmethod
    def _get_default_privacy_keywords() -> List[str]:
        """
        Get comprehensive list of privacy-related keywords in both English and Chinese.

        Categories include:
        - Personal Identification
        - Contact Information
        - Financial Information
        - Medical Information
        - Government Documents
        - Digital Credentials
        - Location Information
        - Employment Information
        """
        return [
            # Personal Identification (English)
            "name", "full name", "first name", "last name", "middle name",
            "date of birth", "dob", "birth date", "birthday",
            "social security", "ssn", "social security number",
            "driver license", "drivers license", "license number",
            "passport", "passport number", "national id",

            # Personal Identification (Chinese)
            "姓名", "全名", "名字", "姓氏", "中间名",
            "出生日期", "生日", "出生年月日",
            "社会保障号", "社保号", "身份证",
            "驾驶证", "驾照", "驾驶证号",
            "护照", "护照号", "身份证号",

            # Contact Information (English)
            "email", "e-mail", "email address", "mail",
            "phone", "phone number", "mobile", "cell", "telephone",
            "address", "home address", "street address", "zip code", "postal code",
            "contact", "contact info", "contact information",

            # Contact Information (Chinese)
            "邮箱", "电子邮件", "邮件地址",
            "电话", "手机号", "手机", "电话号码", "联系电话",
            "地址", "家庭地址", "街道地址", "邮编", "邮政编码",
            "联系方式", "联系信息",

            # Financial Information (English)
            "credit card", "debit card", "card number", "cvv", "cvc",
            "bank account", "account number", "routing number", "iban", "swift",
            "bank", "banking", "finance", "financial",
            "salary", "income", "wage", "payroll",

            # Financial Information (Chinese)
            "信用卡", "借记卡", "银行卡", "卡号", "cvv", "cvc",
            "银行账户", "账号", "账户号", "路由号", "iban", "swift",
            "银行", "银行业务", "金融", "财务",
            "薪水", "收入", "工资", "薪资",

            # Medical Information (English)
            "medical", "health", "doctor", "physician", "hospital",
            "medication", "prescription", "diagnosis", "treatment",
            "insurance", "health insurance", "medical insurance",

            # Medical Information (Chinese)
            "医疗", "健康", "医生", "医师", "医院",
            "药物", "处方", "诊断", "治疗",
            "保险", "健康保险", "医疗保险",

            # Government Documents (English)
            "tax id", "taxpayer id", "tin", "ein",
            "voter id", "voting registration",
            "immigration", "visa", "citizenship",

            # Government Documents (Chinese)
            "税号", "纳税人号", "tin", "ein",
            "选民证", "投票登记",
            "移民", "签证", "公民身份",

            # Digital Credentials (English)
            "password", "passcode", "pin", "security code",
            "username", "user id", "login", "account",
            "token", "api key", "secret key", "access key",

            # Digital Credentials (Chinese)
            "密码", "验证码", "pin码", "安全码",
            "用户名", "用户id", "登录", "账户",
            "令牌", "api密钥", "秘钥", "访问密钥",

            # Location Information (English)
            "location", "gps", "coordinates", "latitude", "longitude",
            "city", "state", "province", "country",

            # Location Information (Chinese)
            "位置", "gps", "坐标", "纬度", "经度",
            "城市", "州", "省份", "国家",

            # Employment Information (English)
            "employer", "company", "job title", "position",
            "employee id", "staff id", "work history",

            # Employment Information (Chinese)
            "雇主", "公司", "职位", "职称",
            "员工id", "员工号", "工作经历",

            # Biometric Data (English)
            "biometric", "biometrics", "fingerprint", "fingerprints",
            "facial recognition", "face id", "iris scan", "retina scan",
            "voice recognition", "dna", "genetic",

            # Biometric Data (Chinese)
            "生物识别", "指纹", "指纹识别", "面部识别", "虹膜扫描",
            "视网膜扫描", "语音识别", "dna", "基因",

            # Financial Transactions (English)
            "transaction", "transfer", "payment", "deposit", "withdrawal",
            "balance", "statement", "invoice", "receipt", "wire transfer",

            # Financial Transactions (Chinese)
            "交易", "转账", "支付", "存款", "取款", "余额", "对账单",
            "发票", "收据", "电汇",

            # Financial Documents (English)
            "tax return", "w2", "1099", "financial statement", "audit report",
            "payroll", "salary slip", "bonus", "compensation",

            # Financial Documents (Chinese)
            "纳税申报", "w2表", "1099表", "财务报表", "审计报告",
            "工资单", "薪资单", "奖金", "补偿",

            # Medical History (English)
            "medical history", "surgical history", "allergy", "allergies",
            "chronic condition", "chronic disease", "family history",
            "previous surgery", "past treatment",

            # Medical History (Chinese)
            "病史", "手术史", "过敏", "过敏史", "慢性病", "家族史",
            "既往手术", "既往治疗",

            # Prescriptions & Medications (English)
            "prescription", "dosage", "milligrams", "mg", "tablets", "capsules",
            "pills", "injection", "intravenous", "oral medication",

            # Prescriptions & Medications (Chinese)
            "处方", "剂量", "毫克", "mg", "片剂", "胶囊", "药丸", "注射",
            "静脉注射", "口服药",

            # Encryption Keys (English)
            "private key", "public key", "ssl certificate", "tls certificate",
            "rsa key", "aes key", "encryption key", "decryption key",

            # Encryption Keys (Chinese)
            "私钥", "公钥", "ssl证书", "tls证书", "rsa密钥", "aes密钥",
            "加密密钥", "解密密钥",

            # Precise Location (English)
            "coordinates", "latitude", "longitude", "gps location",
            "real-time location", "current position", "exact address",

            # Precise Location (Chinese)
            "坐标", "纬度", "经度", "gps位置", "实时位置", "当前位置",
            "精确地址",

            # Location History (English)
            "location history", "tracking data", "movement pattern",
            "geofence", "check-in", "route", "path", "trajectory",

            # Location History (Chinese)
            "位置历史", "跟踪数据", "移动模式", "地理围栏", "签到",
            "路线", "路径", "轨迹",

            # Email Content (English)
            "email content", "email attachment", "email metadata",
            "email header", "email body", "sent emails", "received emails",

            # Email Content (Chinese)
            "邮件内容", "邮件附件", "邮件元数据", "邮件头", "邮件正文",
            "发送邮件", "接收邮件",

            # Messaging Content (English)
            "chat log", "instant message", "text message", "sms",
            "conversation", "message history", "chat record",

            # Messaging Content (Chinese)
            "聊天记录", "即时消息", "短信", "会话", "消息历史", "聊天记录",

            # Call Logs (English)
            "call log", "call history", "phone call", "voicemail",
            "contact list", "phone book", "dialed numbers",

            # Call Logs (Chinese)
            "通话记录", "通话历史", "电话", "语音邮件", "联系人列表",
            "电话簿", "拨打号码",

            # Social Security (English)
            "social security", "social benefits", "welfare", "unemployment",
            "disability benefits", "retirement benefits",

            # Social Security (Chinese)
            "社会保障", "社会福利", "失业救济", "残疾福利", "退休福利",

            # Relationship Data (English)
            "relationship", "family member", "contact network",
            "social persistence", "personal relationship",

            # Relationship Data (Chinese)
            "关系", "家庭成员", "联系人网络", "社交关系", "个人关系",

            # Behavioral Data (English)
            "behavior pattern", "usage habit", "preference", "setting",
            "browsing history", "search history", "app usage",

            # Behavioral Data (Chinese)
            "行为模式", "使用习惯", "偏好", "设置", "浏览历史",
            "搜索历史", "应用使用",

            # Additional Privacy Terms
            "confidential", "private", "sensitive", "personal data",
            "pii", "personally identifiable information",
            "秘密", "隐私", "敏感", "个人数据",
            "pii", "个人可识别信息"
        ]
