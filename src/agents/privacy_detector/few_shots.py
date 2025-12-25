"""Optimized few-shot examples for Privacy Detector Agent."""

PRIVACY_DETECTOR_FEW_SHOTS = """Here are examples of privacy leak detection in conversation records:

EXAMPLE 1 - Personal Identifiers Leak:
Conversation Records:
[1] USER: Hi, I'm John Smith, born on March 15, 1985. My SSN is 123-45-6789.
[2] ASSISTANT: Hello John! I'll help you with your request.

Analysis Result:
{
  "privacy_detected": true,
  "privacy_leaks": [
    {
      "privacy_type": "personal_identifiers",
      "severity": "high",
      "reasoning": "The conversation contains full name (John Smith), date of birth (March 15, 1985), and Social Security Number (123-45-6789). This combination enables identity theft and fraud.",
      "detected_items": ["Full name: John Smith", "Date of birth: March 15, 1985", "SSN: 123-45-6789"]
    }
  ],
  "overall_severity": "high",
  "reasoning": "Multiple personal identifiers were exposed that could enable identity theft"
}

EXAMPLE 2 - Contact Information Leak:
Conversation Records:
[1] USER: You can reach me at 123 Main Street, Springfield, IL 62701. My phone is (555) 123-4567 and email is john.smith@email.com.
[2] ASSISTANT: Got your contact details. I'll send the information there.

Analysis Result:
{
  "privacy_detected": true,
  "privacy_leaks": [
    {
      "privacy_type": "contact_info",
      "severity": "high",
      "reasoning": "Complete physical address, phone number, and email address provided. This enables unwanted contact, spam, or physical stalking.",
      "detected_items": ["Address: 123 Main Street, Springfield, IL 62701", "Phone: (555) 123-4567", "Email: john.smith@email.com"]
    }
  ],
  "overall_severity": "high",
  "reasoning": "Complete contact information was exposed enabling unwanted communication"
}

EXAMPLE 3 - Financial Accounts Leak:
Conversation Records:
[1] USER: My bank account number is 123456789012 and routing number is 021000021. Credit card ends with 4567.
[2] ASSISTANT: I'll process your banking information securely.

Analysis Result:
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
Conversation Records:
[1] USER: My password is 'Summer2024!' and the API key is sk-1234567890abcdef. The 2FA code is 123456.
[2] ASSISTANT: Your credentials are now configured.

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
Conversation Records:
[1] USER: I've had diabetes since 2015, had bypass surgery in 2018, and I'm allergic to penicillin.
[2] ASSISTANT: I understand your medical background. Here's some relevant information.

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
Conversation Records:
[1] USER: I'm currently at coordinates 40.7128° N, 74.0060° W, which is exactly Times Square in Manhattan.
[2] ASSISTANT: That's a very specific location! What can I help you with?

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
Conversation Records:
[1] USER: I just received an email from my boss about the company layoffs next month. It says 20 positions will be cut.
[2] ASSISTANT: That sounds like sensitive company information. You should be careful sharing that.

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
Conversation Records:
[1] USER: What's the weather like today?
[2] ASSISTANT: It's sunny and 75 degrees. Would you like me to check another location?

Analysis Result:
{
  "privacy_detected": false,
  "privacy_leaks": [],
  "overall_severity": "none",
  "reasoning": "The conversation contains only general weather inquiries without any personal, financial, medical, or sensitive information. No privacy concerns detected."
}

EXAMPLE 9 - Biometric Data Leak:
Conversation Records:
[1] USER: My fingerprint scan registered successfully. The system captured all ten fingerprints and facial recognition data.
[2] ASSISTANT: Your biometric registration is complete.

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
Conversation Records:
[1] USER: I have admin access to the database with username 'db_admin' and the connection string is mongodb://admin:secret123@localhost:27017/company_db
[2] ASSISTANT: I'll help you troubleshoot the database connection.

Analysis Result:
{
  "privacy_detected": true,
  "privacy_leaks": [
    {
      "privacy_type": "system_access",
      "severity": "critical",
      "reasoning": "Database admin credentials and connection string provide complete access to sensitive company data and systems.",
      "detected_items": ["Admin username: db_admin", "Connection string: mongodb://admin:secret123@localhost:27017/company_db"]
    }
  ],
  "overall_severity": "critical",
  "reasoning": "Critical system access credentials were exposed"
}

EXAMPLE 11 - Government IDs Leak:
Conversation Records:
[1] USER: My passport number is P123456789 and driver's license is D987654321. ID card expires in 2026.
[2] ASSISTANT: Your identification documents are verified.

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
Conversation Records:
[1] USER: The JWT token is eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9... and the refresh token is abc123def456.
[2] ASSISTANT: Your authentication tokens are set up.

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
"""
