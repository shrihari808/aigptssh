# CREATE THIS NEW FILE

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from config import REQUIRE_API_KEY, VALID_API_KEY

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def api_key_auth(api_key: str = Security(api_key_header)):
    """
    Dependency to validate the API key.
    Checks the REQUIRE_API_KEY flag before validating.
    """
    if REQUIRE_API_KEY == 1:
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key is required"
            )
        if api_key != VALID_API_KEY:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key"
            )