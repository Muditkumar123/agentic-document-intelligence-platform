# RFC 9110 - HTTP Semantics: Status Code Classes

The status code of a response is a three-digit integer code that describes the result of the request and the semantics of the response. The first digit of the status code defines the class of response.

HTTP status codes are extensible. A client is not required to understand the meaning of all registered status codes, but it must understand the class of any status code, as indicated by the first digit.

| Class | Range | Meaning |
| --- | --- | --- |
| Informational | 1xx | The request was received, continuing process |
| Successful | 2xx | The request was successfully received, understood, and accepted |
| Redirection | 3xx | Further action needs to be taken in order to complete the request |
| Client Error | 4xx | The request contains bad syntax or cannot be fulfilled |
| Server Error | 5xx | The server failed to fulfill an apparently valid request |

The 404 (Not Found) status code indicates that the origin server did not find a current representation for the target resource or is not willing to disclose that one exists.

The 503 (Service Unavailable) status code indicates that the server is currently unable to handle the request due to a temporary overload or scheduled maintenance, which will likely be alleviated after some delay.
