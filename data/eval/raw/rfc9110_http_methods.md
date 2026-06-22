# RFC 9110 - HTTP Semantics: Safe and Idempotent Methods

Request methods are considered safe if their defined semantics are essentially read-only; the client does not request, and does not expect, any state change on the origin server as a result of applying a safe method. The GET, HEAD, OPTIONS, and TRACE methods are defined as safe.

A request method is considered idempotent if the intended effect on the server of multiple identical requests with that method is the same as the effect for a single such request. Of the request methods defined by this specification, PUT, DELETE, and safe request methods are idempotent. Note that POST is neither safe nor idempotent.
