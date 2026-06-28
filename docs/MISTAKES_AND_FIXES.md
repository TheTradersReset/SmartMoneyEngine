# Mistakes and Fixes

## Issue-01

ImportError

Reason:

utils.py had no add() function.

Solution:

Implemented add() correctly.

---

## Issue-02

load_csv() returned None.

Reason:

Function contained only pass.

Solution:

Implemented complete function.

---

## Issue-03

Logger.success()

Reason:

Standard logger has no success().

Solution:

Changed to logger.info().

---

## Issue-04

Missing Exception

Reason:

CSVFileNotFoundError not defined.

Solution:

Implemented custom exceptions.