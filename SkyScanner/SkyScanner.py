import http.client


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
    print(res.code)
    print(res.read().decode('utf-8'))
    print(res.headers["Location"].split('/')[-1])
    # get the session key from the response and return it
    return res.headers["Location"].split('/')[-1]


def poll_session_results(session_key: str):
    conn = http.client.HTTPSConnection("skyscanner-skyscanner-flight-search-v1.p.rapidapi.com")

    headers = {
        'x-rapidapi-host': "skyscanner-skyscanner-flight-search-v1.p.rapidapi.com",
        'x-rapidapi-key': session_key
    }

    # parameters
    parameters = ['sortType=price', 'sortOrder=asc', 'duration=1800', 'pageIndex=0', 'pageSize=100']

    conn.request("GET",
                 f"/apiservices/pricing/uk2/v1.0/{session_key}?{'&'.join(parameters)}",
                 headers=headers)

    res = conn.getresponse()
    data = res.read()

    print(data.decode("utf-8"))


if __name__ == "__main__":
    session_key = get_session_key()
    poll_session_results(session_key)
