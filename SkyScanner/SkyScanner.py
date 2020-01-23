import http.client
from time import sleep
from json import loads
from typing import List

from MachineLearning import try_function


def get_session_key(country='US', currency='USD', locale='en-US', originPlace='SFO-sky',
                    destinationPlace='LHR-sky', outboundDate='2020-01-20', adults='1',
                    inboundDate='2020-01-25', cabinClass='business', children='0', infants='0') -> str:
    """
    Send POST msg to server tog et a session key.
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
    print(res.code)
    print(res.read().decode('utf-8'))
    print(res.headers["Location"].split('/')[-1])
    # get the session key from the response and return it
    return res.headers["Location"].split('/')[-1]


def poll_session_results(session_key: str, sortType='price', sortOrder='asc', duration='1800', pageIndex='0',
                             pageSize='100') -> List[List[str]]:
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
    :return: A list of list of 3 items: price (int), InboundLegId (str), OutboundLegId (str).
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
                 f"/apiservices/pricing/uk2/v1.0/75233e42-4a01-42cb-9ee7-74a1229a45b9?{parameters}",
                 headers=headers)
    # get the response
    res = conn.getresponse()
    # read the response
    data = res.read().decode('utf-8')

    # print(data)

    # convert the response to json
    data_j = loads(data)
    # get the flights
    flights = data_j["Itineraries"]

    # print(json.dumps(data_j, indent=4, sort_keys=True))
    # print(json.dumps(flights, indent=4, sort_keys=True))

    # create the flights properties list
    flights_properties = [[flight["PricingOptions"][0]["Price"], flight["InboundLegId"], flight["OutboundLegId"]] for
                          flight in flights]

    return flights_properties


def default_poll_session_results() -> List[List[str]]:
    key = ""
    while key == "":
        try:
            key = get_session_key()
        except AssertionError:
            print("Failed getting a key. Trying again in 1 sec.")
            sleep(1)
    return poll_session_results(key)


if __name__ == "__main__":
    flights = default_poll_session_results()
    print(flights)
