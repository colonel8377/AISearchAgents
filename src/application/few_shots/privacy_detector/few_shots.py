"""Optimized few-shot examples for Privacy Detector Agent."""

PRIVACY_DETECTOR_FEW_SHOTS = """Here are examples of systematic privacy leak detection in user messages:

EXAMPLE 1 - Comprehensive Identity Leak:
User Messages:
[1] User: Hi, I'm John Smith, born on March 15, 1985. My SSN is 123-45-6789 and I live at 123 Main Street, Springfield, IL 62701. You can reach me at (555)123-4567 or john.smith@email.com.

Analysis Result:
{
  "privacy_detected": true,
  "privacy_leaks": [
    {
      "privacy_type": "personal_identifiers",
      "severity": "critical",
      "reasoning": "Complete identity package: full name, date of birth, SSN, full address, phone, and email. This enables comprehensive identity theft, fraud, and stalking.",
      "detected_items": ["Full name: John Smith", "DOB: March 15, 1985", "SSN: 123-45-6789", "Address: 123 Main Street, Springfield, IL 62701", "Phone: (555)123-4567", "Email: john.smith@email.com"]
    }
  ],
  "overall_severity": "critical",
  "reasoning": "Complete identity compromise with all major identifier categories exposed"
}

EXAMPLE 2 - Technical Credentials Leak:
User Messages:
[1] User: My API key is sk-1234567890abcdef and the persistence password is 'Admin2024!'. The JWT token is eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

Analysis Result:
{
  "privacy_detected": true,
  "privacy_leaks": [
    {
      "privacy_type": "api_keys_tokens",
      "severity": "critical",
      "reasoning": "Active API key and persistence credentials exposed. This enables unauthorized system access, data breaches, and potential ransomware attacks.",
      "detected_items": ["API Key: sk-1234567890abcdef", "Database Password: Admin2024!", "JWT Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."]
    }
  ],
  "overall_severity": "critical",
  "reasoning": "System access credentials exposed enabling unauthorized infrastructure access"
}

EXAMPLE 3 - Pattern-Based Detection:
User Messages:
[1] User: I just got a new credit card ending in 4567 and my bank account is 123456789012 with routing 021000021. Also, my phone is 555.123.4567.

Analysis Result:
{
  "privacy_detected": true,
  "privacy_leaks": [
    {
      "privacy_type": "financial_accounts",
      "severity": "high",
      "reasoning": "Bank account number, routing number, and partial credit card information exposed. Enables financial fraud and unauthorized transactions.",
      "detected_items": ["Bank Account: 123456789012", "Routing Number: 021000021", "Credit Card (partial): *4567"]
    },
    {
      "privacy_type": "contact_info",
      "severity": "medium",
      "reasoning": "Phone number exposed in financial context, potentially enabling fraud verification or contact.",
      "detected_items": ["Phone: 555.123.4567"]
    }
  ],
  "overall_severity": "high",
  "reasoning": "Financial account details exposed enabling direct monetary loss"
}

EXAMPLE 4 - Contextual Risk Assessment:
User Messages:
[1] User: As a patient, my medical record number is MRN-12345 and I've been diagnosed with diabetes. My doctor's name is Dr. Johnson at City Hospital.

Analysis Result:
{
  "privacy_detected": true,
  "privacy_leaks": [
    {
      "privacy_type": "health_records",
      "severity": "high",
      "reasoning": "Medical diagnosis and doctor information exposed in healthcare context. While some sharing may be necessary, this creates privacy risks for medical identity theft.",
      "detected_items": ["Diagnosis: diabetes", "Doctor: Dr. Johnson", "Hospital: City Hospital", "MRN: MRN-12345"]
    }
  ],
  "overall_severity": "high",
  "reasoning": "Medical information exposed in healthcare communication context"
}

EXAMPLE 5 - No Privacy Risk:
User Messages:
[1] User: I like sunny weather and prefer coffee over tea. What's the weather like today?

Analysis Result:
{
  "privacy_detected": false,
  "privacy_leaks": [],
  "overall_severity": "none",
  "reasoning": "Message contains only general preferences and weather inquiry. No sensitive personal information detected."
}
{
  "privacy_detected": true,
  "privacy_leaks": [
    {
      "privacy_type": "financial_accounts",
      "severity": "critical",
      "reasoning": "Direct exposure of bank account details and partial credit card information creates immediate risk of financial fraud and unauthorized transactions.",
      "detected_items": ["Bank account: 123456789012", "Routing number: 021000021", "Credit card (last 4): 4567"]
    }
  ],
  "overall_severity": "critical",
  "reasoning": "Critical financial account information was exposed"
}

EXAMPLE 4 - Authentication Credentials Leak:
User Messages:
[1] User: My password is 'Summer2024!' and the API key is sk-1234567890abcdef. The 2FA code is 123456.

Analysis Result:
{
  "privacy_detected": true,
  "privacy_leaks": [
    {
      "privacy_type": "authentication_credentials",
      "severity": "critical",
      "reasoning": "Direct exposure of password, API key, and 2FA code enables complete account takeover and unauthorized access to all associated services.",
      "detected_items": ["Password: Summer2024!", "API key: sk-1234567890abcdef", "2FA code: 123456"]
    }
  ],
  "overall_severity": "critical",
  "reasoning": "Critical authentication credentials were exposed"
}

EXAMPLE 5 - Medical History Leak:
User Messages:
[1] User: I've had diabetes since 2015, had bypass surgery in 2018, and I'm allergic to penicillin.
[2] User: I understand your medical background. Here's some relevant information.

Analysis Result:
{
  "privacy_detected": true,
  "privacy_leaks": [
    {
      "privacy_type": "medical_history",
      "severity": "medium",
      "reasoning": "Detailed medical history including chronic condition (diabetes), surgical procedures (bypass surgery), and allergies (penicillin) could impact insurance, employment, or social perceptions.",
      "detected_items": ["Chronic condition: diabetes (since 2015)", "Surgical history: bypass surgery (2018)", "Allergy: penicillin"]
    }
  ],
  "overall_severity": "medium",
  "reasoning": "Sensitive medical history information was exposed"
}

EXAMPLE 6 - Precise Location Leak:
User Messages:
[1] User: I'm currently at coordinates 40.7128° N, 74.0060° W, which is exactly Times Square in Manhattan.
[2] User: That's a very specific location! What can I help you with?

Analysis Result:
{
  "privacy_detected": true,
  "privacy_leaks": [
    {
      "privacy_type": "precise_location",
      "severity": "medium",
      "reasoning": "Precise GPS coordinates combined with location identification could enable real-time tracking or targeted location-based harassment.",
      "detected_items": ["GPS coordinates: 40.7128° N, 74.0060° W", "Identified location: Times Square, Manhattan"]
    }
  ],
  "overall_severity": "medium",
  "reasoning": "Precise location information was exposed"
}

EXAMPLE 7 - Email Content Leak:
User Messages:
[1] User: I just received an email from my boss about the company layoffs next month. It says 20 positions will be cut.
[2] User: That sounds like sensitive company information. You should be careful sharing that.

Analysis Result:
{
  "privacy_detected": true,
  "privacy_leaks": [
    {
      "privacy_type": "email_content",
      "severity": "medium",
      "reasoning": "Sharing confidential email content about company layoffs could breach confidentiality agreements and create workplace issues.",
      "detected_items": ["Confidential email content: company layoffs", "Specific details: 20 positions to be cut"]
    }
  ],
  "overall_severity": "medium",
  "reasoning": "Confidential email content was shared inappropriately"
}

EXAMPLE 8 - No Privacy Leak:
User Messages:
[1] User: What's the weather like today?
[2] User: It's sunny and 75 degrees. Would you like me to check another location?

Analysis Result:
{
  "privacy_detected": false,
  "privacy_leaks": [],
  "overall_severity": "none",
  "reasoning": "The conversation contains only general weather inquiries without any personal, financial, medical, or sensitive information. No privacy concerns detected."
}

EXAMPLE 9 - Biometric Data Leak:
User Messages:
[1] User: My fingerprint scan registered successfully. The system captured all ten fingerprints and facial recognition data.
[2] User: Your biometric registration is complete.

Analysis Result:
{
  "privacy_detected": true,
  "privacy_leaks": [
    {
      "privacy_type": "biometric_data",
      "severity": "high",
      "reasoning": "Biometric data including fingerprints and facial recognition patterns are extremely sensitive and cannot be changed if compromised.",
      "detected_items": ["Fingerprint data: all ten fingerprints", "Facial recognition data: captured"]
    }
  ],
  "overall_severity": "high",
  "reasoning": "Highly sensitive biometric data was exposed"
}

EXAMPLE 10 - System Access Leak:
User Messages:
[1] User: I have admin access to the persistence with username 'db_admin' and the persistence string is mongodb://admin:secret123@localhost:27017/company_db
[2] User: I'll help you troubleshoot the persistence persistence.

Analysis Result:
{
  "privacy_detected": true,
  "privacy_leaks": [
    {
      "privacy_type": "system_access",
      "severity": "critical",
      "reasoning": "Database admin credentials and persistence string provide complete access to sensitive company data and systems.",
      "detected_items": ["Admin username: db_admin", "Connection string: mongodb://admin:secret123@localhost:27017/company_db"]
    }
  ],
  "overall_severity": "critical",
  "reasoning": "Critical system access credentials were exposed"
}

EXAMPLE 11 - Government IDs Leak:
User Messages:
[1] User: My passport number is P123456789 and driver's license is D987654321. ID card expires in 2026.
[2] User: Your identification documents are verified.

Analysis Result:
{
  "privacy_detected": true,
  "privacy_leaks": [
    {
      "privacy_type": "government_ids",
      "severity": "high",
      "reasoning": "Government-issued identification numbers (passport, driver's license) can be used for identity theft and accessing government services.",
      "detected_items": ["Passport number: P123456789", "Driver's license: D987654321", "ID expiration: 2026"]
    }
  ],
  "overall_severity": "high",
  "reasoning": "Government-issued identification documents were exposed"
}

EXAMPLE 12 - API Keys and Tokens Leak:
User Messages:
[1] User: The JWT token is eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9... and the refresh token is abc123def456.
[2] User: Your authentication tokens are set up.

Analysis Result:
{
  "privacy_detected": true,
  "privacy_leaks": [
    {
      "privacy_type": "api_keys_tokens",
      "severity": "critical",
      "reasoning": "JWT and refresh tokens provide ongoing access to authenticated systems and services until they expire.",
      "detected_items": ["JWT token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...", "Refresh token: abc123def456"]
    }
  ],
  "overall_severity": "critical",
  "reasoning": "Critical API tokens and authentication credentials were exposed"
}

EXAMPLE 13 - Real Privacy Data (Chinese):
User Messages:
[1] User: 我的手机号是13812345678，邮箱zhangsan@company.com

Analysis Result:
{
  "privacy_detected": true,
  "privacy_leaks": [
    {
      "privacy_type": "PHONE_NUMBER_CN",
      "severity": "high",
      "severity_level": "L3",
      "detected_items": ["13812345678"],
      "confidence": 0.95,
      "reasoning": "中国手机号格式正确，用户主动提供个人信息"
    },
    {
      "privacy_type": "EMAIL_ADDRESS",
      "severity": "high",
      "severity_level": "L3",
      "detected_items": ["zhangsan@company.com"],
      "confidence": 0.92,
      "reasoning": "邮箱格式正确，出现在个人分享上下文中"
    }
  ],
  "overall_severity": "high",
  "overall_severity_level": "L3"
}

EXAMPLE 14 - Example/Test Data (Should Filter):
User Messages:
[1] User: 请输入手机号，格式如：13800138000

Analysis Result:
{
  "privacy_detected": false,
  "privacy_leaks": [],
  "reasoning": "号码出现在'格式如'说明文字中，是示例而非真实数据"
}

EXAMPLE 15 - Version Number (False Positive):
User Messages:
[1] User: 服务器版本 v1.2.3.456

Analysis Result:
{
  "privacy_detected": false,
  "privacy_leaks": [],
  "reasoning": "'版本'上下文明确表明这是软件版本号，不是电话号码"
}

EXAMPLE 16 - Credential in Code Context:
User Messages:
[1] User: 配置文件中 api_key = 'sk-1234567890abcdef1234567890abcdef1234567890abcdef'

Analysis Result:
{
  "privacy_detected": true,
  "privacy_leaks": [
    {
      "privacy_type": "OPENAI_KEY",
      "severity": "critical",
      "severity_level": "L4",
      "detected_items": ["sk-1234567890abcdef1234567890abcdef1234567890abcdef"],
      "confidence": 0.98,
      "reasoning": "OpenAI API Key 格式 (sk-前缀)，出现在配置上下文中"
    }
  ],
  "overall_severity": "critical",
  "overall_severity_level": "L4"
}

EXAMPLE 17 - Mixed Real and Example:
User Messages:
[1] User: 我的身份证是110101199001010018，不是示例的000000000000000000

Analysis Result:
{
  "privacy_detected": true,
  "privacy_leaks": [
    {
      "privacy_type": "CN_ID_CARD",
      "severity": "critical",
      "severity_level": "L4",
      "detected_items": ["110101199001010018"],
      "confidence": 0.96,
      "reasoning": "用户明确区分真实数据和示例，110101199001010018校验位正确"
    }
  ],
  "note": "000000000000000000 被正确识别为示例数据并排除"
}

GUIDELINES:
- Consider context and potential impact of information exposure
- Be conservative - err on the side of caution for ambiguous cases
- Assess real-world harm potential (identity theft, fraud, harassment, system compromise, etc.)
- Look for combinations of information that increase risk
- Consider both direct statements and contextual implications
- Evaluate severity based on potential consequences, not just information type
- Different privacy types may have different severity implications
- Biometric and authentication data are typically high/critical severity
- Location and communication data may be medium depending on context
- Financial and system access data are usually critical severity
- IMPORTANT: Distinguish between real data and example/test data
- Example indicators: "格式如", "for example", "e.g.", sequential digits (123456789), test patterns (555-xxx, test@example.com)
- Real data indicators: Personal context ("我的", "my"), non-pattern values, conversation flow suggests real sharing
"""
