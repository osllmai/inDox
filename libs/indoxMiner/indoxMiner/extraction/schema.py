from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import json
from .fields import (
    ValidationPatterns,
    ValidationRule,
    CommonValidationRules,
    Field,
    FieldType,
)


@dataclass
class ExtractorSchema:
    """Schema definition for data extraction with validation and formatting.

    Attributes:
        fields (List[Field]): List of fields to extract
        examples (List[Dict[str, Any]], optional): Example extractions
        context (str, optional): Additional context for extraction
    """

    fields: List[Field]
    examples: Optional[List[Dict[str, Any]]] = None
    context: Optional[str] = None

    def to_prompt(self, text: str) -> str:
        """Generate extraction prompt based on schema.

        Args:
            text (str): Source text for extraction

        Returns:
            str: Formatted prompt for LLM extraction
        """
        # Describe each field in the schema for extraction instructions
        fields_desc = "\n".join(
            f"- {field.to_prompt_string()}" for field in self.fields
        )

        # Include any additional context if provided
        context_section = f"\nContext:\n{self.context}\n" if self.context else ""

        # Add examples, if available, for model guidance
        examples_section = ""
        if self.examples:
            examples_json = json.dumps(self.examples, indent=2)
            examples_section = f"\nExamples:\n{examples_json}\n"

        # Construct the prompt with detailed requirements and nested structure emphasis
        return f"""Task: Extract structured information from the given text according to the following schema.

    Fields to extract:
    {fields_desc}{context_section}{examples_section}

    Output Requirements:
    1. Extract ONLY the specified fields as defined in the schema.
    2. Follow exact field names and types, including nested structures, lists, and dictionaries as specified.
    3. Use JSON format (default).
    4. Retain any nested lists or dictionaries as defined in the schema, maintaining structure in JSON format.
    5. If a required field cannot be found, use null/empty values.
    6. Validate all values based on provided rules within the schema.
    7. For dates, use ISO format (YYYY-MM-DD).
    8. Return ONLY the output in JSON format - no additional text, comments, or explanations.
    9. Critical: Do not alter field structure, even for nested or list fields.

    Text to analyze:
    {text}"""


class Schema:
    Passport = ExtractorSchema(
        fields=[
            Field(
                name="Passport Number",
                description="Unique passport number",
                field_type=FieldType.STRING,
                required=True,
                rules=CommonValidationRules.PASSPORT_RULE,
            ),
            Field(
                name="Given Names",
                description="First and middle names",
                field_type=FieldType.STRING,
                required=True,
                rules=CommonValidationRules.NAME_RULE,
            ),
            Field(
                name="Surname",
                description="Family name/surname",
                field_type=FieldType.STRING,
                required=True,
                rules=CommonValidationRules.NAME_RULE,
            ),
            Field(
                name="Date of Birth",
                description="Date of birth in YYYY-MM-DD format",
                field_type=FieldType.DATE,
                required=True,
                rules=CommonValidationRules.DATE_RULE,
            ),
            Field(
                name="Place of Birth",
                description="City and country of birth",
                field_type=FieldType.STRING,
                required=True,
                rules=ValidationRule(min_length=2, max_length=100),
            ),
            Field(
                name="Nationality",
                description="Country of nationality",
                field_type=FieldType.STRING,
                required=True,
                rules=ValidationRule(min_length=2, max_length=30),
            ),
            Field(
                name="Gender",
                description="Gender as specified in passport",
                field_type=FieldType.STRING,
                required=True,
                rules=CommonValidationRules.create_enum_rule(["M", "F", "X"]),
            ),
            Field(
                name="Date of Issue",
                description="Issue date of the passport",
                field_type=FieldType.DATE,
                required=True,
                rules=CommonValidationRules.DATE_RULE,
            ),
            Field(
                name="Date of Expiry",
                description="Expiry date of the passport",
                field_type=FieldType.DATE,
                required=True,
                rules=CommonValidationRules.DATE_RULE,
            ),
            Field(
                name="Place of Issue",
                description="Place where passport was issued",
                field_type=FieldType.STRING,
                required=True,
                rules=ValidationRule(min_length=2, max_length=100),
            ),
            Field(
                name="MRZ",
                description="Machine Readable Zone text",
                field_type=FieldType.STRING,
                rules=ValidationRule(pattern=r"^[A-Z0-9<]{88}$"),
            ),
        ]
    )

    Invoice = ExtractorSchema(
        fields=[
            Field(
                name="Invoice Number",
                description="Unique invoice identifier",
                field_type=FieldType.INTEGER,
                required=True,
            ),
            Field(
                name="Date",
                description="Invoice date",
                field_type=FieldType.DATE,
                required=True,
                rules=CommonValidationRules.DATE_RULE,
            ),
            Field(
                name="Company Name",
                description="Name of the company issuing invoice",
                field_type=FieldType.STRING,
                required=True,
                rules=ValidationRule(min_length=2, max_length=100),
            ),
            Field(
                name="Address",
                description="Address of the company or ship to",
                field_type=FieldType.STRING,
                required=True,
                rules=CommonValidationRules.ADDRESS_RULE,
            ),
            Field(
                name="Customer Name",
                description="Name of the customer",
                field_type=FieldType.STRING,
                required=True,
                rules=CommonValidationRules.NAME_RULE,
            ),
            Field(
                name="Items",
                description="List of items in the invoice",
                field_type=FieldType.LIST,
                array_item_type=FieldType.DICT,
                dict_fields={
                    "description": FieldType.STRING,
                    "quantity": FieldType.FLOAT,
                    "unit_price": FieldType.FLOAT,
                    "total": FieldType.FLOAT,
                },
            ),
            Field(
                name="Subtotal",
                description="Sum before tax",
                field_type=FieldType.FLOAT,
                required=True,
                rules=CommonValidationRules.AMOUNT_RULE,
            ),
            Field(
                name="Tax Amount",
                description="Total tax amount",
                field_type=FieldType.FLOAT,
                required=True,
                rules=CommonValidationRules.AMOUNT_RULE,
            ),
            Field(
                name="Total Amount",
                description="Final total amount including tax",
                field_type=FieldType.FLOAT,
                required=True,
                rules=CommonValidationRules.AMOUNT_RULE,
            ),
        ]
    )

    Flight_Ticket = ExtractorSchema(
        fields=[
            Field(
                name="Ticket Number",
                description="Unique ticket identifier",
                field_type=FieldType.STRING,
                required=True,
                rules=ValidationRule(pattern=r"^\d{13}$"),
            ),
            Field(
                name="Passenger Name",
                description="Full name of passenger",
                field_type=FieldType.STRING,
                required=True,
                rules=CommonValidationRules.NAME_RULE,
            ),
            Field(
                name="Flight Number",
                description="Airline flight number",
                field_type=FieldType.STRING,
                required=True,
                rules=CommonValidationRules.FLIGHT_RULE,
            ),
            Field(
                name="Departure Airport",
                description="Airport of departure",
                field_type=FieldType.STRING,
                required=True,
                rules=ValidationRule(pattern=ValidationPatterns.IATA_AIRPORT),
            ),
            Field(
                name="Arrival Airport",
                description="Airport of arrival",
                field_type=FieldType.STRING,
                required=True,
                rules=ValidationRule(pattern=ValidationPatterns.IATA_AIRPORT),
            ),
            Field(
                name="Departure DateTime",
                description="Date and time of departure",
                field_type=FieldType.DATE,
                required=True,
                rules=CommonValidationRules.DATE_RULE,
            ),
            Field(
                name="Arrival DateTime",
                description="Date and time of arrival",
                field_type=FieldType.DATE,
                required=True,
                rules=CommonValidationRules.DATE_RULE,
            ),
            Field(
                name="Seat Number",
                description="Assigned seat",
                field_type=FieldType.STRING,
                rules=ValidationRule(pattern=ValidationPatterns.SEAT_NUMBER),
            ),
            Field(
                name="Class",
                description="Travel class",
                field_type=FieldType.STRING,
                required=True,
                rules=CommonValidationRules.create_enum_rule(
                    ["Economy", "Business", "First"]
                ),
            ),
            Field(
                name="Booking Reference",
                description="PNR or booking reference",
                field_type=FieldType.STRING,
                required=True,
                rules=CommonValidationRules.BOOKING_RULE,
            ),
            Field(
                name="Fare",
                description="Ticket fare amount",
                field_type=FieldType.FLOAT,
                required=True,
                rules=CommonValidationRules.AMOUNT_RULE,
            ),
        ]
    )

    Bank_Statement = ExtractorSchema(
        fields=[
            Field(
                name="Account Holder",
                description="Name of the account holder",
                field_type=FieldType.STRING,
                required=True,
                rules=CommonValidationRules.NAME_RULE,
            ),
            Field(
                name="Account Number",
                description="Bank account number",
                field_type=FieldType.STRING,
                required=True,
                rules=ValidationRule(
                    pattern=ValidationPatterns.BANK_ACCOUNT["generic"]
                ),
            ),
            Field(
                name="IBAN",
                description="International Bank Account Number",
                field_type=FieldType.STRING,
                rules=ValidationRule(pattern=ValidationPatterns.BANK_ACCOUNT["iban"]),
            ),
            Field(
                name="Statement Period",
                description="Statement period (YYYY-MM)",
                field_type=FieldType.STRING,
                required=True,
                rules=ValidationRule(pattern=r"^\d{4}-(?:0[1-9]|1[0-2])$"),
            ),
            Field(
                name="Opening Balance",
                description="Balance at start of period",
                field_type=FieldType.FLOAT,
                required=True,
                rules=CommonValidationRules.AMOUNT_RULE,
            ),
            Field(
                name="Closing Balance",
                description="Balance at end of period",
                field_type=FieldType.FLOAT,
                required=True,
                rules=CommonValidationRules.AMOUNT_RULE,
            ),
            Field(
                name="Transactions",
                description="List of transactions",
                field_type=FieldType.LIST,
                array_item_type=FieldType.DICT,
                dict_fields={
                    "date": FieldType.DATE,
                    "description": FieldType.STRING,
                    "amount": FieldType.FLOAT,
                    "type": FieldType.STRING,
                    "reference": FieldType.STRING,
                },
            ),
        ]
    )

    Medical_Record = ExtractorSchema(
        fields=[
            Field(
                name="Patient Name",
                description="Full name of the patient",
                field_type=FieldType.STRING,
                required=True,
                rules=CommonValidationRules.NAME_RULE,
            ),
            Field(
                name="Date of Birth",
                description="Patient's date of birth",
                field_type=FieldType.DATE,
                required=True,
                rules=CommonValidationRules.DATE_RULE,
            ),
            Field(
                name="Medical Record Number",
                description="Unique medical record identifier",
                field_type=FieldType.STRING,
                required=True,
                rules=ValidationRule(pattern=r"^MRN-\d{8}$"),
            ),
            Field(
                name="Patient ID",
                description="National healthcare ID",
                field_type=FieldType.STRING,
                rules=ValidationRule(pattern=r"^\d{10}$"),
            ),
            Field(
                name="Diagnosis",
                description="Medical diagnosis",
                field_type=FieldType.LIST,
                array_item_type=FieldType.DICT,
                dict_fields={
                    "code": FieldType.STRING,  # ICD code
                    "description": FieldType.STRING,
                    "date": FieldType.DATE,
                },
            ),
            Field(
                name="Medications",
                description="Prescribed medications",
                field_type=FieldType.LIST,
                array_item_type=FieldType.DICT,
                dict_fields={
                    "name": FieldType.STRING,
                    "dosage": FieldType.STRING,
                    "frequency": FieldType.STRING,
                    "start_date": FieldType.DATE,
                    "end_date": FieldType.DATE,
                },
            ),
            Field(
                name="Physician",
                description="Treating physician information",
                field_type=FieldType.DICT,
                dict_fields={
                    "name": FieldType.STRING,
                    "license": FieldType.STRING,
                    "specialty": FieldType.STRING,
                },
            ),
            Field(
                name="Vital Signs",
                description="Patient vital signs",
                field_type=FieldType.DICT,
                dict_fields={
                    "blood_pressure": FieldType.STRING,
                    "heart_rate": FieldType.INTEGER,
                    "temperature": FieldType.FLOAT,
                    "respiratory_rate": FieldType.INTEGER,
                },
            ),
        ]
    )

    Driver_License = ExtractorSchema(
        fields=[
            Field(
                name="License Number",
                description="Driver's license number",
                field_type=FieldType.STRING,
                required=True,
                rules=ValidationRule(pattern=r"^[A-Z0-9]{6,12}$"),
            ),
            Field(
                name="Full Name",
                description="Full name of license holder",
                field_type=FieldType.STRING,
                required=True,
                rules=CommonValidationRules.NAME_RULE,
            ),
            Field(
                name="Address",
                description="Residential address",
                field_type=FieldType.STRING,
                required=True,
                rules=CommonValidationRules.ADDRESS_RULE,
            ),
            Field(
                name="Date of Birth",
                description="Date of birth",
                field_type=FieldType.DATE,
                required=True,
                rules=CommonValidationRules.DATE_RULE,
            ),
            Field(
                name="Issue Date",
                description="License issue date",
                field_type=FieldType.DATE,
                required=True,
                rules=CommonValidationRules.DATE_RULE,
            ),
            Field(
                name="Expiry Date",
                description="License expiry date",
                field_type=FieldType.DATE,
                required=True,
                rules=CommonValidationRules.DATE_RULE,
            ),
            Field(
                name="Class",
                description="License class/type",
                field_type=FieldType.STRING,
                required=True,
                rules=CommonValidationRules.create_enum_rule(["A", "B", "C", "D", "E"]),
            ),
            Field(
                name="Restrictions",
                description="License restrictions",
                field_type=FieldType.LIST,
                array_item_type=FieldType.STRING,
            ),
            Field(
                name="Endorsements",
                description="License endorsements",
                field_type=FieldType.LIST,
                array_item_type=FieldType.STRING,
            ),
        ]
    )
    Resume = ExtractorSchema(
        fields=[
            Field(
                name="Full Name",
                description="Full name of the person",
                field_type=FieldType.STRING,
                required=True,
                rules=CommonValidationRules.NAME_RULE,
            ),
            Field(
                name="Contact",
                description="Contact information",
                field_type=FieldType.DICT,
                required=True,
                dict_fields={
                    "email": FieldType.STRING,
                    "phone": FieldType.STRING,
                    "address": FieldType.STRING,
                    "linkedin": FieldType.STRING,
                    "portfolio": FieldType.STRING,
                    "github": FieldType.STRING,
                },
            ),
            Field(
                name="Professional Summary",
                description="Brief professional summary or objective statement",
                field_type=FieldType.STRING,
                rules=ValidationRule(min_length=50, max_length=500),
            ),
            # Work Experience
            Field(
                name="Work Experience",
                description="Professional work experience",
                field_type=FieldType.LIST,
                required=True,
                array_item_type=FieldType.DICT,
                dict_fields={
                    "company": FieldType.STRING,
                    "position": FieldType.STRING,
                    "location": FieldType.STRING,
                    "start_date": FieldType.DATE,
                    "end_date": FieldType.DATE,
                    "current": FieldType.BOOLEAN,
                    "responsibilities": FieldType.LIST,
                    "achievements": FieldType.LIST,
                    "technologies": FieldType.LIST,
                },
            ),
            # Education
            Field(
                name="Education",
                description="Educational background",
                field_type=FieldType.LIST,
                required=True,
                array_item_type=FieldType.DICT,
                dict_fields={
                    "institution": FieldType.STRING,
                    "degree": FieldType.STRING,
                    "field_of_study": FieldType.STRING,
                    "location": FieldType.STRING,
                    "start_date": FieldType.DATE,
                    "end_date": FieldType.DATE,
                    "gpa": FieldType.FLOAT,
                    "honors": FieldType.LIST,
                    "relevant_courses": FieldType.LIST,
                },
            ),
            # Skills
            Field(
                name="Skills",
                description="Professional skills grouped by category",
                field_type=FieldType.DICT,
                dict_fields={
                    "technical": FieldType.LIST,
                    "soft": FieldType.LIST,
                    "languages": FieldType.LIST,
                    "tools": FieldType.LIST,
                    "certifications": FieldType.LIST,
                },
            ),
            # Projects
            Field(
                name="Projects",
                description="Notable projects",
                field_type=FieldType.LIST,
                array_item_type=FieldType.DICT,
                dict_fields={
                    "name": FieldType.STRING,
                    "description": FieldType.STRING,
                    "technologies": FieldType.LIST,
                    "url": FieldType.STRING,
                    "start_date": FieldType.DATE,
                    "end_date": FieldType.DATE,
                    "highlights": FieldType.LIST,
                },
            ),
            # Awards & Achievements
            Field(
                name="Awards",
                description="Notable awards and achievements",
                field_type=FieldType.LIST,
                array_item_type=FieldType.DICT,
                dict_fields={
                    "title": FieldType.STRING,
                    "issuer": FieldType.STRING,
                    "date": FieldType.DATE,
                    "description": FieldType.STRING,
                },
            ),
            # Certifications
            Field(
                name="Certifications",
                description="Professional certifications",
                field_type=FieldType.LIST,
                array_item_type=FieldType.DICT,
                dict_fields={
                    "name": FieldType.STRING,
                    "issuer": FieldType.STRING,
                    "date_earned": FieldType.DATE,
                    "expiry_date": FieldType.DATE,
                    "credential_id": FieldType.STRING,
                    "url": FieldType.STRING,
                },
            ),
            # Publications
            Field(
                name="Publications",
                description="Academic or professional publications",
                field_type=FieldType.LIST,
                array_item_type=FieldType.DICT,
                dict_fields={
                    "title": FieldType.STRING,
                    "authors": FieldType.LIST,
                    "publication": FieldType.STRING,
                    "date": FieldType.DATE,
                    "url": FieldType.STRING,
                    "description": FieldType.STRING,
                },
            ),
            # Volunteer Experience
            Field(
                name="Volunteer Experience",
                description="Volunteer work and community service",
                field_type=FieldType.LIST,
                array_item_type=FieldType.DICT,
                dict_fields={
                    "organization": FieldType.STRING,
                    "role": FieldType.STRING,
                    "start_date": FieldType.DATE,
                    "end_date": FieldType.DATE,
                    "description": FieldType.STRING,
                    "achievements": FieldType.LIST,
                },
            ),
            # Additional Information
            Field(
                name="Additional Information",
                description="Other relevant information",
                field_type=FieldType.DICT,
                dict_fields={
                    "interests": FieldType.LIST,
                    "conferences": FieldType.LIST,
                    "memberships": FieldType.LIST,
                    "references": FieldType.LIST,
                },
            ),
        ],
    )
