"""The HTTP contract with the served model.

What may be sent and how the answer is read (:mod:`sampling`).

Nothing here talks to a GPU or imports a model framework: this is the client
side, and it must work on a machine that could not possibly run the model.

A server is addressed by its base URL and nothing else. There is no endpoint
record file to look one up in, so whatever holds the URL — an environment
variable, a command-line flag — is the whole of the addressing story.
"""
