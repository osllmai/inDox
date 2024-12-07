
class MapsTextSearch:
    """
    Search for an address using OpenStreetMap's Nominatim API and handle printing.

    Parameters:
    - user_agent (str): The user agent string to be used by the Nominatim geolocator. Default is "osm_search".

    Methods:
    - search_address(address: str) -> None: Search for the given address and print the details.

    Notes:
    - The class uses the Geopy library to interact with the OpenStreetMap Nominatim API.
    - Results, including the full address, latitude, and longitude, are printed.
    """


    def __init__(self, user_agent: str = "osm_search"):
        """
        Initialize the MapsTextSearch with the specified user agent.

        Parameters:
        - user_agent (str): The user agent for the Nominatim geolocator. Default is "osm_search".
        """
        from geopy.geocoders import Nominatim
        from geopy.exc import GeopyError
        try:
            self.geolocator = Nominatim(user_agent=user_agent)
            print(f"MapsTextSearch initialized with user agent: {user_agent}")
        except GeopyError as e:
            print(f"Error initializing geolocator: {e}")
        except Exception as e:
            print(f"Unexpected error during initialization: {e}")

    def search_address(self, address: str) -> None:
        """
        Search for the given address and print the details.

        Parameters:
        - address (str): The address to search for.

        Returns:
        - None

        Notes:
        - Prints the full address, latitude, and longitude if the address is found.
        - Prints a message if the address is not found.
        - Prints an error message if an exception occurs during the search process.
        """
        from geopy.exc import GeopyError

        try:
            location = self.geolocator.geocode(address)
            if location:
                output = (f"Address: {location.address}\n"
                          f"Latitude: {location.latitude}, Longitude: {location.longitude}")
                print(output)
            else:
                print("Address not found")
        except GeopyError as e:
            print(f"Geopy error occurred: {e}")
        except Exception as e:
            print(f"Unexpected error occurred while searching for the address: {e}")
            self._log_fallback(f"Unexpected error: {e}")

    def _log_fallback(self, error_message: str) -> None:
        """
        Attempt to log an error to a fallback location if an unexpected error occurs.

        Parameters:
        - error_message (str): The error message to be logged.

        Returns:
        - None

        Notes:
        - Appends the error message to a fallback log file.
        """
        try:
            with open('fallback_log.txt', 'a') as fallback_log:
                fallback_log.write(f"{error_message}\n")
        except Exception as fallback_error:
            print(f"Error occurred while logging the original error: {fallback_error}")

