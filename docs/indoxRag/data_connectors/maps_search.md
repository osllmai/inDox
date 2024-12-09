---

# MapsTextSearch

## Overview

The `MapsTextSearch` class allows users to search for addresses using the OpenStreetMap Nominatim API. It handles address searching, logging, and printing results. It uses the Geopy library to interact with the Nominatim API.

## Installation

Ensure you have the necessary library installed:

```bash
pip install geopy
```

## Quick Start

1. **Initialize the `MapsTextSearch` Class**

   To use the Nominatim API, you need to provide a user agent string. The user agent should be a meaningful name or identifier that helps Nominatim understand the source of the requests.

   **Creating a User Agent:**

   - **Purpose:** The user agent helps identify your application to the Nominatim service. It should be unique to your application or project.
   - **Format:** The user agent string should ideally include your application name and a contact email or website for any potential follow-up.

   **Example User Agents:**

   - `"my_application/1.0 (contact@example.com)"`
   - `"my_project_name (http://myprojectwebsite.com)"`

   **Initialize the Class:**

   ```python
   from indoxRag.data_connectors import MapsTextSearch

   # Initialize MapsTextSearch with a custom user agent
   searcher = MapsTextSearch(user_agent="my_application/1.0 (contact@example.com)")
   ```

2. **Search for an Address**

   - Call the `search_address` method with the address you want to search for.

   ```python
   address = "Pizza Tower, Via Lecco, 20124 Milan, Italy"
   searcher.search_address(address)
   ```

## Class `MapsTextSearch`

### Methods

#### `__init__(self, user_agent: str = "osm_search")`

Initializes the `MapsTextSearch` object with the specified user agent.

**Parameters:**

- `user_agent` (str): The user agent for the Nominatim geolocator. Default is `"osm_search"`. It should be a unique identifier for your application or project.

**Notes:**

- Configures logging to capture and format log messages.
- Handles logging configuration errors by printing a message.

#### `search_address(self, address: str) -> None`

Searches for the given address and prints and logs the details.

**Parameters:**

- `address` (str): The address to search for.

**Notes:**

- If the address is found, it prints and logs the full address, latitude, and longitude.
- If the address is not found, it prints and logs a "Address not found" message.
- If an error occurs during the search process, it prints and logs the error message, and attempts to log it to a fallback location.

### Error Handling

- Logs any issues encountered during the address search or logging processes.
- Uses a fallback logging mechanism to handle errors during the primary logging process.

## Example Usage

```python
from indoxRag.data_connectors import MapsTextSearch

# Initialize MapsTextSearch object with a user agent
searcher = MapsTextSearch(user_agent="my_application/1.0 (contact@example.com)")

# Search for an address
address = "Pizza Tower, Via Lecco, 20124 Milan, Italy"
searcher.search_address(address)
```
