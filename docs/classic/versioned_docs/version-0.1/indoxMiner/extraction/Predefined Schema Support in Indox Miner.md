# Predefined Schema Support

Indox Miner supports a variety of **predefined schemas** to streamline and standardize the process of extracting structured information from documents. These schemas act as templates that define which fields to extract, along with validation rules to ensure data consistency and accuracy.

## Purpose of Predefined Schemas

Predefined schemas help:

- **Standardize data extraction** across different document types.
- **Simplify the setup** for users by providing ready-to-use configurations.
- **Ensure data quality** by applying field-specific validation rules.
- **Support different document types** (e.g., invoices, passports, medical records) with minimal configuration.

## Available Predefined Schemas

Indox Miner includes the following predefined schemas:

### 1. Passport Schema

Extracts key information from passport documents, including:

- **Passport Number**: Unique passport identifier.
- **Given Names**: First and middle names.
- **Surname**: Family name/surname.
- **Date of Birth**: Formatted as YYYY-MM-DD.
- **Place of Birth**: Location of birth (city and country).
- **Nationality**: Country of citizenship.
- **Gender**: Gender identifier (M/F/X).
- **Date of Issue**: Passport issue date.
- **Date of Expiry**: Passport expiration date.
- **Place of Issue**: Location of passport issuance.
- **MRZ**: Machine-readable zone text.

### 2. Invoice Schema

Extracts data from invoices, focusing on financial and transactional details:

- **Invoice Number**: Unique invoice identifier.
- **Date**: Invoice issue date.
- **Company Name**: Name of issuing company.
- **Address**: Address associated with the company or billing.
- **Customer Name**: Name of the customer.
- **Items**: List of items with description, quantity, unit price, and total.
- **Subtotal**: Pre-tax amount.
- **Tax Amount**: Total tax amount.
- **Total Amount**: Grand total, including tax.

### 3. Flight Ticket Schema

Captures essential details from flight tickets, such as:

- **Ticket Number**: Unique ticket identifier.
- **Passenger Name**: Full name of the passenger.
- **Flight Number**: Airline flight number.
- **Departure Airport** and **Arrival Airport**: IATA codes for origin and destination airports.
- **Departure DateTime** and **Arrival DateTime**: Departure and arrival times.
- **Seat Number**: Assigned seat.
- **Class**: Travel class (Economy, Business, First).
- **Booking Reference**: PNR or booking code.
- **Fare**: Ticket fare.

### 4. Bank Statement Schema

Supports extraction from bank statements with fields like:

- **Account Holder**: Name on the account.
- **Account Number**: Bank account number.
- **IBAN**: International Bank Account Number.
- **Statement Period**: Month and year of statement.
- **Opening Balance** and **Closing Balance**: Account balances.
- **Transactions**: List of transactions with date, description, amount, type, and reference.

### 5. Medical Record Schema

Facilitates the extraction of patient medical records:

- **Patient Name**: Full name of the patient.
- **Date of Birth**: Formatted as YYYY-MM-DD.
- **Medical Record Number**: Unique identifier for the record.
- **Diagnosis**: List of diagnoses with code, description, and date.
- **Medications**: List of prescribed medications.
- **Physician**: Information about the treating physician.
- **Vital Signs**: Basic vitals like blood pressure, heart rate, temperature.

### 6. Driver License Schema

Extracts information from driver licenses, such as:

- **License Number**: Unique identifier.
- **Full Name**: License holder’s full name.
- **Address**: Residential address.
- **Date of Birth**: Formatted as YYYY-MM-DD.
- **Issue Date** and **Expiry Date**: License issuance and expiration dates.
- **Class**: License class or type.
- **Restrictions** and **Endorsements**: Lists of restrictions and endorsements.

### 7. Resume Schema

Enables structured extraction from resumes or CVs:

- **Full Name**: Candidate's name.
- **Contact**: Includes email, phone, address, LinkedIn, portfolio, GitHub.
- **Professional Summary**: Short summary or objective.
- **Work Experience**: List of work history with company, position, dates, and achievements.
- **Education**: Academic background, degrees, institutions, dates, and GPA.
- **Skills**: Categorized technical, soft, and language skills.
- **Projects**: Description of key projects.
- **Awards**, **Certifications**, **Publications**, **Volunteer Experience**, **Additional Information**.

## How to Use Predefined Schemas

To use a predefined schema:

1. Select the appropriate schema based on the document type.
2. Configure the extractor to use the selected schema.
3. Run the extraction, and Indox Miner will apply the schema's structure and validation rules.

## Customization and Extension

While Indox Miner provides these predefined schemas out of the box, users can:

- **Customize existing schemas** by modifying field requirements or validation rules.
- **Create new schemas** for unsupported document types by defining custom fields and rules.

## Conclusion

Predefined schemas in Indox Miner make it easier and faster to extract structured information from various documents accurately and consistently. By leveraging these schemas, users can significantly reduce the setup time required for data extraction tasks and ensure high-quality data output.
