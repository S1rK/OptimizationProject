import http.client
from time import sleep
import json
from numpy import array
from typing import List
import ast

# attributes = ["Price", "Departure Time", "Arrival Time", "Flight Duration", "Number of stops"]
attributes = ["Price", "Flight Duration", "Number of stops"]

def get_session_key(country='US', currency='USD', locale='en-US', originPlace='SFO-sky',
                    destinationPlace='LHR-sky', outboundDate='2020-05-01', adults='1',
                    inboundDate='2020-05-10', cabinClass='business', children='0', infants='0') -> str:
    """
    Sends POST msg to server to get a session key.
    MUST
    :param country: The market/country your user is in (see docs for list of markets).
    :param currency: The currency you want the prices in (3-letter currency code).
    :param locale: The locale you want the results in (ISO locale).
    :param originPlace: The origin place (see docs for places).
    :param destinationPlace: The destination place (see docs for places).
    :param outboundDate: The outbound date. Format “yyyy-mm-dd”.
    :param adults: Number of adults (16+ years). Must be between 1 and 8.
    OPTIONAL
    :param inboundDate: The return date. Format “yyyy-mm-dd”. Use empty string for oneway trip.
    :param cabinClass: The cabin class. Can be “economy”, “premiumeconomy”, “business”, “first”.
    :param children: Number of children (1-16 years). Can be between 0 and 8.
    :param infants: Number of infants (under 12 months). Can be between 0 and 8.
    RETURN
    :return: a session key (str).
    """""
    # create a connection
    conn = http.client.HTTPSConnection("skyscanner-skyscanner-flight-search-v1.p.rapidapi.com")

    # the request's payload
    payload = ['inboundDate=' + inboundDate, 'cabinClass=' + cabinClass, 'children=' + children, 'infants=' + infants,
               'country=' + country, 'currency=' + currency, 'locale=' + locale, 'originPlace=' + originPlace,
               'destinationPlace=' + destinationPlace, 'outboundDate=' + outboundDate, 'adults=' + adults]
    payload = "&".join(payload)
    # the request's headers
    headers = {
        'x-rapidapi-host': "skyscanner-skyscanner-flight-search-v1.p.rapidapi.com",
        'x-rapidapi-key': "f4bcd8cb76msh78693b656187c5ap1f670bjsn7ae9c61a6b51",
        'content-type': "application/x-www-form-urlencoded"
        }
    # send the request
    conn.request("POST", "/apiservices/pricing/v1.0", payload, headers)
    # read the response
    res = conn.getresponse()
    # print the response
    assert res.code == 201, "Problem with getting a session key:\n"+res.read().decode('utf-8')
    # print(res.code)
    # print(res.read().decode('utf-8'))
    # print(res.headers["Location"].split('/')[-1])
    # get the session key from the response and return it
    return res.headers["Location"].split('/')[-1]


def poll_session_results(session_key: str, sortType='price', sortOrder='asc', duration='1800', pageIndex='0',
                             pageSize='100') -> List[List[float]]:
    """
    Sends a GET Poll session results request which returns a {pagesize} number of flights, sorted by {sortType} in
    {sortOrder} order.
    :param session_key: The session key received in the Location Header when creating the session.
    :param sortType: The parameter to sort results on. Can be carrier, duration, outboundarrivetime, outbounddeparttime,
                     inboundarrivetime, inbounddeparttime, price*.
    :param sortOrder: The sort order. ‘asc’ or 'desc’.
    :param duration: Filter for maximum duration in minutes. Integer between 0 and 1800.
    :param pageIndex: The desired page number. Leave empty for no pagination.
    :param pageSize: The number of itineraries per page. Defaults to 10 if not specified.
    :return: A vector (np.array) of fights. Each flight is a vector (np.array) of 5 items:
             minimum price of the flight (float), departure time (str "yyyy-mm-ddThh:mm:ss"),
             arrival time (str "yyyy-mm-ddThh:mm:ss"), duration of the flight (int), number of stops (int)
    """
    # create a connection
    conn = http.client.HTTPSConnection("skyscanner-skyscanner-flight-search-v1.p.rapidapi.com")

    # the request's headers
    headers = {
        'x-rapidapi-host': "skyscanner-skyscanner-flight-search-v1.p.rapidapi.com",
        'x-rapidapi-key': "f4bcd8cb76msh78693b656187c5ap1f670bjsn7ae9c61a6b51"
    }

    # parameters
    parameters = ['sortType=' + sortType, 'sortOrder=' + sortOrder, 'duration=' + duration, 'pageIndex=' + pageIndex,
                  'pageSize=' + pageSize]
    parameters = '&'.join(parameters)
    # send the request
    conn.request("GET",
                 f"/apiservices/pricing/uk2/v1.0/{session_key}?{parameters}",
                 headers=headers)
    # get the response
    res = conn.getresponse()
    # read the response
    data = res.read().decode('utf-8')

    # make sure we got HTTP 200 OK response
    assert res.code == 200, "Problem with getting a session's results:\n"+data

    # convert the response to json
    data_j = json.loads(data)

    # print(json.dumps(data_j, indent=4, sort_keys=True))

    # get the flights
    flights = data_j["Itineraries"]
    legs = data_j["Legs"]

    # create the flights properties list
    # flights_properties = array([array(
    #         [float(flight["PricingOptions"][0]["Price"]), leg["Departure"], leg["Arrival"], int(leg["Duration"]),
    #          int(len(leg["Stops"]))]) for flight, leg in zip(flights, legs)])

    flights_properties = [
        [float(flight["PricingOptions"][0]["Price"]), float(leg["Duration"]), float(len(leg["Stops"]))] for flight, leg
        in zip(flights, legs)]

    return flights_properties


def get_flights(numberOfFlights=100, country='US', currency='USD', locale='en-US', originPlace='SFO-sky',
                         destinationPlace='LHR-sky', outboundDate='2020-05-01', adults='1',
                         inboundDate='2020-05-10', cabinClass='business', children='0', infants='0',
                         sortType='price', sortOrder='asc', duration='1800', pageIndex='0',
                         pageSize='1000') -> array:
    """
    Sends a POST msg to server to get a session key (with the given parameters). Then, sends a GET Poll session results
    request, and returns the flights in a vector.

    REQUIRED for getting a session key
    :param country: The market/country your user is in (see docs for list of markets).
    :param currency: The currency you want the prices in (3-letter currency code).
    :param locale: The locale you want the results in (ISO locale).
    :param originPlace: The origin place (see docs for places).
    :param destinationPlace: The destination place (see docs for places).
    :param outboundDate: The outbound date. Format “yyyy-mm-dd”.
    :param adults: Number of adults (16+ years). Must be between 1 and 8.

    OPTIONAL for getting a session key
    :param inboundDate: The return date. Format “yyyy-mm-dd”. Use empty string for oneway trip.
    :param cabinClass: The cabin class. Can be “economy”, “premiumeconomy”, “business”, “first”.
    :param children: Number of children (1-16 years). Can be between 0 and 8.
    :param infants: Number of infants (under 12 months). Can be between 0 and 8.


    OPTIONAL for getting the session's results
    :param sortType: The parameter to sort results on. Can be carrier, duration, outboundarrivetime, outbounddeparttime,
                     inboundarrivetime, inbounddeparttime, price*.
    :param sortOrder: The sort order. ‘asc’ or 'desc’.
    :param duration: Filter for maximum duration in minutes. Integer between 0 and 1800.
    :param pageIndex: The desired page number. Leave empty for no pagination.
    :param pageSize: The number of itineraries per page. Defaults to 10 if not specified.


    :return: A vector (np.array) of fights. Each flight is a vector (np.array) of 5 items:
             minimum price of the flight (float), departure time (str "yyyy-mm-ddThh:mm:ss"),
             arrival time (str "yyyy-mm-ddThh:mm:ss"), duration of the flight (int), number of stops (int)
    """
    # getting the session's key
    key = ""
    while key == "":
        try:
            key = get_session_key(country=country, currency=currency, locale=locale, originPlace=originPlace,
                                  destinationPlace=destinationPlace, outboundDate=outboundDate, adults=adults,
                                  inboundDate=inboundDate, cabinClass=cabinClass, children=children, infants=infants)
            print(f"got a key: {key}.")
        except AssertionError as e:
            print(str(e))
            print("Failed getting a key. Trying again in 1 sec.")
            print("------------------------------------------------------------------------------------------------")
            sleep(1)

    # getting the session's results
    results = None
    while results is None:
        try:
            results = poll_session_results(session_key=key, sortType=sortType, sortOrder=sortOrder, duration=duration,
                                        pageIndex=pageIndex, pageSize=pageSize)
            print(f"got {len(results)} flights")
        except AssertionError as e:
            print(str(e))
            print("Failed getting session's results. Trying again.")
            print("------------------------------------------------------------------------------------------------")

    # convert all the flights (as lists of floats) to strings
    results = [str(x) for x in results]

    # convert the list of flights to a set of flights, to get rid of all the duplicates
    s = set(results)

    # convert back the flights from strings to lists, and then to np.array, and return an np.array of them
    results = array([array(ast.literal_eval(x)) for x in s])

    return results


def flight_to_string(flight: array) -> str:
    flight = [f"{attributes[i]}: {flight[i]}" for i in range(len(attributes))]
    return ", ".join(flight)


if __name__ == "__main__":
    # learn(0)
    # flights_ = get_flights()
    # print(flights_)

    # fs = [[0.5, 0.3, 0.8], [0.5, 0.3, 0.8], [0.22, 0.354, 0.45], [0.5, 0.3, 0.8]]
    # s = set()
    # for f in fs:
    #     f_str = str(f)
    #     s.add(f_str)
    #     print(f"current different flights: {len(s)}")
    #
    # yay = array([array(ast.literal_eval(x)) for x in s])
    #
    # print(yay)
    # print(type(yay))
    # print(type(yay[0]))
    # print(type(yay[0][0]))

    # fs = get_flights(sortType="", sortOrder="")
    fs = get_flights()
    print(fs)
    print(len(fs))
    print(type(fs))
    print(type(fs[0]))
    print(type(fs[0][0]))

