#!/usr/bin/env python3
"""
Cloudflare MCP Server - DNS and Zone Management for Claude Code.

Provides tools to manage Cloudflare DNS records and zones via the Cloudflare API.
Enables Claude Code to automate DNS configuration for domains.

Requirements:
    - Cloudflare API token with Zone:Read and DNS:Edit permissions

Credential Resolution (in order):
    1. OpenBao Agent (if available at http://127.0.0.1:18200)
    2. CLOUDFLARE_API_TOKEN environment variable

Environment variables:
    CLOUDFLARE_API_TOKEN: API token from Cloudflare dashboard
    OPENBAO_AGENT_ADDR: (Optional) Agent address, defaults to http://127.0.0.1:18200
"""

import os
import sys
import json
from typing import Optional, List, Literal
from enum import Enum

import httpx
from pydantic import BaseModel, Field, ConfigDict, field_validator
from mcp.server.fastmcp import FastMCP

# OpenBao Agent Configuration
AGENT_ADDR = os.getenv("OPENBAO_AGENT_ADDR", "http://127.0.0.1:18200")
AGENT_TIMEOUT = float(os.getenv("OPENBAO_AGENT_TIMEOUT", "5.0"))
DEV_MODE = os.getenv("OPENBAO_DEV_MODE", "").lower() in ("1", "true", "yes")

# Arc Forge secret path configuration
ARC_CLIENT = os.getenv("ARC_CLIENT", "client0")
ARC_ENVIRONMENT = os.getenv("ARC_ENVIRONMENT", "prod")
ARC_USERNAME = os.getenv("ARC_USERNAME", "samuelrodda")

# Initialize MCP server
mcp = FastMCP("cloudflare_mcp")

# Constants
API_BASE_URL = "https://api.cloudflare.com/client/v4"
CHARACTER_LIMIT = 25000
DEFAULT_PER_PAGE = 50


# =============================================================================
# OpenBao Integration
# =============================================================================


class OpenBaoError(Exception):
    """Base exception for OpenBao errors."""
    pass


class AgentNotRunningError(OpenBaoError):
    """Raised when the OpenBao Agent is not running."""
    pass


class SecretNotFoundError(OpenBaoError):
    """Raised when a secret path doesn't exist."""
    pass


def _get_openbao_client():
    """Get HTTP client for OpenBao agent communication."""
    return httpx.Client(
        base_url=AGENT_ADDR,
        timeout=AGENT_TIMEOUT,
        headers={"X-Vault-Request": "true"}
    )


def _get_secret_from_agent(path: str) -> dict:
    """
    Read a secret from the OpenBao Agent.

    Args:
        path: Secret path (e.g., "client0/prod-mcp-cloudflare-samuelrodda")

    Returns:
        The secret data dict.

    Raises:
        AgentNotRunningError: If agent is not running
        SecretNotFoundError: If secret path doesn't exist
        OpenBaoError: For other OpenBao errors
    """
    path = path.lstrip("/")
    full_path = f"/v1/secret/data/{path}"

    try:
        with _get_openbao_client() as client:
            response = client.get(full_path)

            if response.status_code == 404:
                raise SecretNotFoundError(f"Secret not found: {path}")

            if response.status_code != 200:
                raise OpenBaoError(
                    f"Failed to read secret: {response.status_code} - {response.text}"
                )

            data = response.json()
            return data.get("data", {}).get("data", {})

    except httpx.ConnectError:
        raise AgentNotRunningError(
            f"Cannot connect to OpenBao Agent at {AGENT_ADDR}. "
            "Start the agent with:\n"
            "  export BW_SESSION=$(bw unlock --raw)\n"
            "  start-openbao-mcp"
        )


def _build_cloudflare_secret_path() -> str:
    """
    Build secret path for Cloudflare using Arc Forge pattern.

    Pattern: {client}/{environment}-mcp-cloudflare-{username}

    Cloudflare is user-scoped for personal Cloudflare account access.

    Returns:
        Secret path string (e.g., "client0/prod-mcp-cloudflare-samuelrodda")
    """
    return f"{ARC_CLIENT}/{ARC_ENVIRONMENT}-mcp-cloudflare-{ARC_USERNAME}"


# Enums
class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"


class DNSRecordType(str, Enum):
    """Supported DNS record types."""
    A = "A"
    AAAA = "AAAA"
    CNAME = "CNAME"
    MX = "MX"
    TXT = "TXT"
    NS = "NS"
    SRV = "SRV"
    CAA = "CAA"
    PTR = "PTR"


# Pydantic Input Models
class ListZonesInput(BaseModel):
    """Input for listing Cloudflare zones."""
    model_config = ConfigDict(str_strip_whitespace=True)

    name: Optional[str] = Field(
        default=None,
        description="Filter by zone name (e.g., 'example.com')",
        max_length=253
    )
    page: int = Field(default=1, description="Page number", ge=1)
    per_page: int = Field(default=DEFAULT_PER_PAGE, description="Results per page", ge=1, le=100)
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format: 'markdown' for human-readable or 'json' for machine-readable"
    )


class GetZoneInput(BaseModel):
    """Input for getting a specific zone."""
    model_config = ConfigDict(str_strip_whitespace=True)

    zone: str = Field(
        ...,
        description="Zone name (e.g., 'example.com') or zone ID",
        min_length=1,
        max_length=253
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )


class ListDNSRecordsInput(BaseModel):
    """Input for listing DNS records in a zone."""
    model_config = ConfigDict(str_strip_whitespace=True)

    zone: str = Field(
        ...,
        description="Zone name (e.g., 'example.com') or zone ID",
        min_length=1
    )
    type: Optional[DNSRecordType] = Field(
        default=None,
        description="Filter by record type (A, AAAA, CNAME, MX, TXT, etc.)"
    )
    name: Optional[str] = Field(
        default=None,
        description="Filter by record name (e.g., 'www' or 'mail.example.com')",
        max_length=255
    )
    page: int = Field(default=1, description="Page number", ge=1)
    per_page: int = Field(default=DEFAULT_PER_PAGE, description="Results per page", ge=1, le=100)
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )


class CreateDNSRecordInput(BaseModel):
    """Input for creating a DNS record."""
    model_config = ConfigDict(str_strip_whitespace=True)

    zone: str = Field(
        ...,
        description="Zone name (e.g., 'example.com') or zone ID",
        min_length=1
    )
    type: DNSRecordType = Field(
        ...,
        description="DNS record type (A, AAAA, CNAME, MX, TXT, etc.)"
    )
    name: str = Field(
        ...,
        description="Record name (e.g., '@' for root, 'www', 'mail'). Use '@' or zone name for apex.",
        min_length=1,
        max_length=255
    )
    content: str = Field(
        ...,
        description="Record content (IP address for A/AAAA, hostname for CNAME/MX, text for TXT)",
        min_length=1
    )
    ttl: int = Field(
        default=1,
        description="TTL in seconds. Use 1 for 'automatic' (Cloudflare default)",
        ge=1,
        le=86400
    )
    priority: Optional[int] = Field(
        default=None,
        description="Priority for MX records (required for MX, lower = higher priority)",
        ge=0,
        le=65535
    )
    proxied: bool = Field(
        default=False,
        description="Enable Cloudflare proxy (orange cloud). Only for A/AAAA/CNAME."
    )
    comment: Optional[str] = Field(
        default=None,
        description="Comment for the DNS record",
        max_length=500
    )

    @field_validator('name')
    @classmethod
    def normalize_name(cls, v: str) -> str:
        return v.strip()

    @field_validator('proxied')
    @classmethod
    def validate_proxied(cls, v: bool, info) -> bool:
        # Proxied only works for A, AAAA, CNAME
        record_type = info.data.get('type')
        if v and record_type and record_type not in [DNSRecordType.A, DNSRecordType.AAAA, DNSRecordType.CNAME]:
            raise ValueError(f"Proxied can only be enabled for A, AAAA, or CNAME records, not {record_type}")
        return v


class UpdateDNSRecordInput(BaseModel):
    """Input for updating a DNS record."""
    model_config = ConfigDict(str_strip_whitespace=True)

    zone: str = Field(..., description="Zone name or zone ID", min_length=1)
    record_id: str = Field(..., description="DNS record ID to update", min_length=1)
    type: Optional[DNSRecordType] = Field(default=None, description="New record type")
    name: Optional[str] = Field(default=None, description="New record name", max_length=255)
    content: Optional[str] = Field(default=None, description="New record content")
    ttl: Optional[int] = Field(default=None, description="New TTL in seconds", ge=1, le=86400)
    priority: Optional[int] = Field(default=None, description="New priority for MX records", ge=0, le=65535)
    proxied: Optional[bool] = Field(default=None, description="Enable/disable Cloudflare proxy")
    comment: Optional[str] = Field(default=None, description="New comment", max_length=500)


class DeleteDNSRecordInput(BaseModel):
    """Input for deleting a DNS record."""
    model_config = ConfigDict(str_strip_whitespace=True)

    zone: str = Field(..., description="Zone name or zone ID", min_length=1)
    record_id: str = Field(..., description="DNS record ID to delete", min_length=1)


# Redirect Rules (Single Redirects — http_request_dynamic_redirect phase)
_REDIRECT_STATUS_CODES = (301, 302, 303, 307, 308)


class ListRedirectRulesInput(BaseModel):
    """Input for listing a zone's redirect (Single Redirect) rules."""
    model_config = ConfigDict(str_strip_whitespace=True)

    zone: str = Field(..., description="Zone name (e.g. 'cbrc.events') or zone ID", min_length=1)
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN, description="Output format"
    )


class CreateRedirectRuleInput(BaseModel):
    """Input for creating a redirect (Single Redirect) rule."""
    model_config = ConfigDict(str_strip_whitespace=True)

    zone: str = Field(..., description="Zone name or zone ID", min_length=1)
    target_url: str = Field(
        ..., description="Destination URL, e.g. 'https://cbrcevents.net'", min_length=1
    )
    status_code: int = Field(default=301, description="Redirect status (301/302/303/307/308)")
    preserve_query_string: bool = Field(
        default=True, description="Keep the incoming query string on the redirect"
    )
    preserve_path: bool = Field(
        default=True,
        description="Append the incoming path to target_url (dynamic redirect via concat). "
                    "False = static redirect to target_url verbatim."
    )
    match_expression: Optional[str] = Field(
        default=None,
        description="Cloudflare Rules expression for what to match, e.g. '(http.host eq \"cbrc.events\")'. "
                    "If omitted and source_hostname is given, it's built for you."
    )
    source_hostname: Optional[str] = Field(
        default=None,
        description="Convenience: hostname to match (builds '(http.host eq \"<hostname>\")') when "
                    "match_expression is omitted.",
        max_length=253
    )
    description: Optional[str] = Field(default=None, description="Rule description", max_length=500)
    enabled: bool = Field(default=True, description="Whether the rule is active")

    @field_validator('status_code')
    @classmethod
    def validate_status(cls, v: int) -> int:
        if v not in _REDIRECT_STATUS_CODES:
            raise ValueError(f"status_code must be one of {_REDIRECT_STATUS_CODES}")
        return v


class UpdateRedirectRuleInput(BaseModel):
    """Input for updating a redirect rule by id (only provided fields change)."""
    model_config = ConfigDict(str_strip_whitespace=True)

    zone: str = Field(..., description="Zone name or zone ID", min_length=1)
    rule_id: str = Field(..., description="Redirect rule ID (from cloudflare_list_redirect_rules)", min_length=1)
    target_url: Optional[str] = Field(default=None, description="New destination URL")
    status_code: Optional[int] = Field(default=None, description="New status (301/302/303/307/308)")
    preserve_query_string: Optional[bool] = Field(default=None, description="Keep query string")
    preserve_path: Optional[bool] = Field(default=None, description="Append incoming path to target")
    match_expression: Optional[str] = Field(default=None, description="New match expression")
    source_hostname: Optional[str] = Field(default=None, description="New source hostname to match", max_length=253)
    description: Optional[str] = Field(default=None, description="New description", max_length=500)
    enabled: Optional[bool] = Field(default=None, description="Enable/disable the rule")

    @field_validator('status_code')
    @classmethod
    def validate_status(cls, v):
        if v is not None and v not in _REDIRECT_STATUS_CODES:
            raise ValueError(f"status_code must be one of {_REDIRECT_STATUS_CODES}")
        return v


class DeleteRedirectRuleInput(BaseModel):
    """Input for deleting a redirect rule by id."""
    model_config = ConfigDict(str_strip_whitespace=True)

    zone: str = Field(..., description="Zone name or zone ID", min_length=1)
    rule_id: str = Field(..., description="Redirect rule ID to delete", min_length=1)


# Cloudflare Pages (account-scoped — needs Account -> Cloudflare Pages -> Edit)
class ListPagesProjectsInput(BaseModel):
    """Input for listing Pages projects."""
    model_config = ConfigDict(str_strip_whitespace=True)

    account: Optional[str] = Field(
        default=None, description="Account name or ID (optional if the token sees exactly one account)"
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class GetPagesProjectInput(BaseModel):
    """Input for getting one Pages project."""
    model_config = ConfigDict(str_strip_whitespace=True)

    project_name: str = Field(..., description="Pages project name", min_length=1)
    account: Optional[str] = Field(default=None, description="Account name or ID")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class CreatePagesProjectInput(BaseModel):
    """Input for creating a Pages project (direct-upload, or git-connected to GitHub)."""
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., description="Project name (becomes the <name>.pages.dev subdomain)", min_length=1, max_length=58)
    production_branch: str = Field(default="main", description="Production branch")
    github_owner: Optional[str] = Field(
        default=None,
        description="GitHub owner/org for a git-connected project. Requires the 'Cloudflare Workers and Pages' GitHub App to already be authorized for this account in the dashboard (one-time)."
    )
    github_repo: Optional[str] = Field(default=None, description="GitHub repository name (with github_owner)")
    build_command: Optional[str] = Field(default=None, description="Build command, e.g. 'npm run build'")
    destination_dir: Optional[str] = Field(default=None, description="Build output directory, e.g. 'dist' or 'build'")
    root_dir: Optional[str] = Field(default=None, description="Repo subdirectory to build from (monorepos)")
    account: Optional[str] = Field(default=None, description="Account name or ID")


class DeletePagesProjectInput(BaseModel):
    """Input for deleting a Pages project."""
    model_config = ConfigDict(str_strip_whitespace=True)

    project_name: str = Field(..., description="Pages project name", min_length=1)
    account: Optional[str] = Field(default=None, description="Account name or ID")


class ListPagesDeploymentsInput(BaseModel):
    """Input for listing a Pages project's deployments."""
    model_config = ConfigDict(str_strip_whitespace=True)

    project_name: str = Field(..., description="Pages project name", min_length=1)
    account: Optional[str] = Field(default=None, description="Account name or ID")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class CreatePagesDeploymentInput(BaseModel):
    """Input for triggering a new Pages deployment (git-connected projects)."""
    model_config = ConfigDict(str_strip_whitespace=True)

    project_name: str = Field(..., description="Pages project name", min_length=1)
    branch: Optional[str] = Field(default=None, description="Branch to deploy (defaults to the production branch)")
    account: Optional[str] = Field(default=None, description="Account name or ID")


# Shared utilities
def _is_openbao_agent_available() -> bool:
    """Check if OpenBao agent is reachable."""
    try:
        with _get_openbao_client() as client:
            response = client.get("/v1/sys/health")
            return response.status_code in (200, 429, 472, 473, 501, 503)
    except Exception:
        return False


def _get_api_token() -> str:
    """
    Get Cloudflare API token from OpenBao agent or environment variable.

    Credential Resolution (in order):
    1. OpenBao Agent (if available)
    2. CLOUDFLARE_API_TOKEN environment variable

    Returns:
        API token string.

    Raises:
        ValueError: If API token cannot be retrieved.
    """
    # Try OpenBao first if agent is available
    if _is_openbao_agent_available():
        try:
            secret_path = _build_cloudflare_secret_path()
            secret_data = _get_secret_from_agent(secret_path)
            api_token = secret_data.get("api_token")

            if api_token:
                return api_token
        except OpenBaoError:
            pass  # Fall through to env var

    # Fallback to environment variable
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")
    if api_token:
        return api_token

    raise ValueError(
        "Cloudflare API token not found.\n"
        "Set CLOUDFLARE_API_TOKEN environment variable.\n\n"
        "Create API token at: https://dash.cloudflare.com/profile/api-tokens\n"
        "Required permissions: Zone:Read and DNS:Edit"
    )


def _get_headers() -> dict:
    """Get headers for Cloudflare API requests."""
    return {
        "Authorization": f"Bearer {_get_api_token()}",
        "Content-Type": "application/json"
    }


async def _make_request(
    method: str,
    endpoint: str,
    params: Optional[dict] = None,
    json_data: Optional[dict] = None
) -> dict:
    """Make authenticated request to Cloudflare API."""
    async with httpx.AsyncClient() as client:
        response = await client.request(
            method,
            f"{API_BASE_URL}/{endpoint}",
            headers=_get_headers(),
            params=params,
            json=json_data,
            timeout=30.0
        )
        response.raise_for_status()
        return response.json()


def _handle_error(e: Exception) -> str:
    """Format error messages for tool responses."""
    # Check for credential/OpenBao errors first (before generic errors)
    if isinstance(e, ValueError) and ("openbao" in str(e).lower() or "token" in str(e).lower() or "agent" in str(e).lower()):
        # Pass through the detailed ValueError from _get_api_token()
        return f"Error: {str(e)}"

    if isinstance(e, httpx.HTTPStatusError):
        try:
            error_data = e.response.json()
            errors = error_data.get("errors", [])
            if errors:
                error_msgs = [f"{err.get('code', 'unknown')}: {err.get('message', 'Unknown error')}" for err in errors]
                return f"Cloudflare API Error: {'; '.join(error_msgs)}"
        except Exception:
            pass

        status = e.response.status_code
        if status == 400:
            return "Error: Bad request. Check your parameters."
        elif status == 401:
            return "Error: Authentication failed. Check your CLOUDFLARE_API_TOKEN."
        elif status == 403:
            return "Error: Permission denied. Your API token may lack required permissions."
        elif status == 404:
            return "Error: Resource not found. Check the zone name or record ID."
        elif status == 429:
            return "Error: Rate limit exceeded. Please wait before retrying."
        return f"Error: API request failed with status {status}"
    elif isinstance(e, httpx.TimeoutException):
        return "Error: Request timed out. Please try again."
    elif isinstance(e, ValueError):
        return f"Error: {str(e)}"
    return f"Error: {type(e).__name__}: {str(e)}"


async def _resolve_zone_id(zone: str) -> str:
    """Resolve zone name to zone ID if necessary."""
    # If it looks like a zone ID (32 hex chars), return as-is
    if len(zone) == 32 and all(c in '0123456789abcdef' for c in zone.lower()):
        return zone

    # Otherwise, look up by name
    data = await _make_request("GET", "zones", params={"name": zone})
    zones = data.get("result", [])
    if not zones:
        raise ValueError(f"Zone '{zone}' not found. Check the domain name.")
    return zones[0]["id"]


def _format_zone_markdown(zone: dict) -> str:
    """Format zone data as markdown."""
    lines = [
        f"## {zone['name']}",
        f"- **ID**: `{zone['id']}`",
        f"- **Status**: {zone['status']}",
        f"- **Plan**: {zone.get('plan', {}).get('name', 'Unknown')}",
        f"- **Name Servers**: {', '.join(zone.get('name_servers', []))}",
        f"- **Paused**: {'Yes' if zone.get('paused') else 'No'}",
    ]
    if zone.get('activated_on'):
        lines.append(f"- **Activated**: {zone['activated_on']}")
    return "\n".join(lines)


def _format_record_markdown(record: dict) -> str:
    """Format DNS record as markdown."""
    proxy_status = ""
    if record.get('proxiable'):
        proxy_status = f" (Proxied: {'Yes' if record.get('proxied') else 'No'})"

    ttl_display = "Auto" if record.get('ttl') == 1 else f"{record.get('ttl')}s"

    lines = [
        f"### {record['type']} - {record['name']}",
        f"- **Content**: `{record['content']}`",
        f"- **TTL**: {ttl_display}{proxy_status}",
        f"- **ID**: `{record['id']}`",
    ]
    if record.get('priority') is not None:
        lines.insert(2, f"- **Priority**: {record['priority']}")
    if record.get('comment'):
        lines.append(f"- **Comment**: {record['comment']}")
    return "\n".join(lines)


# Tool definitions
@mcp.tool(
    name="cloudflare_list_zones",
    annotations={
        "title": "List Cloudflare Zones",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def cloudflare_list_zones(params: ListZonesInput) -> str:
    """List all zones (domains) in your Cloudflare account.

    Returns zone names, IDs, status, and nameservers. Use zone IDs for other operations.

    Args:
        params: ListZonesInput containing optional filters and pagination

    Returns:
        List of zones with IDs needed for DNS operations
    """
    try:
        query_params = {
            "page": params.page,
            "per_page": params.per_page
        }
        if params.name:
            query_params["name"] = params.name

        data = await _make_request("GET", "zones", params=query_params)
        zones = data.get("result", [])
        result_info = data.get("result_info", {})

        if not zones:
            return "No zones found in your Cloudflare account."

        if params.response_format == ResponseFormat.JSON:
            return json.dumps({
                "zones": zones,
                "total": result_info.get("total_count", len(zones)),
                "page": result_info.get("page", params.page),
                "per_page": result_info.get("per_page", params.per_page)
            }, indent=2)

        # Markdown format
        lines = ["# Cloudflare Zones", ""]
        lines.append(f"Showing {len(zones)} of {result_info.get('total_count', len(zones))} zones")
        lines.append("")

        for zone in zones:
            lines.append(_format_zone_markdown(zone))
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="cloudflare_get_zone",
    annotations={
        "title": "Get Zone Details",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def cloudflare_get_zone(params: GetZoneInput) -> str:
    """Get details for a specific Cloudflare zone by name or ID.

    Args:
        params: GetZoneInput with zone name or ID

    Returns:
        Zone details including ID, status, nameservers, and plan
    """
    try:
        zone_id = await _resolve_zone_id(params.zone)
        data = await _make_request("GET", f"zones/{zone_id}")
        zone = data.get("result", {})

        if params.response_format == ResponseFormat.JSON:
            return json.dumps(zone, indent=2)

        return _format_zone_markdown(zone)

    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="cloudflare_list_dns_records",
    annotations={
        "title": "List DNS Records",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def cloudflare_list_dns_records(params: ListDNSRecordsInput) -> str:
    """List DNS records for a zone. Filter by type or name.

    Args:
        params: ListDNSRecordsInput with zone and optional filters

    Returns:
        List of DNS records with types, names, content, and IDs
    """
    try:
        zone_id = await _resolve_zone_id(params.zone)

        query_params = {
            "page": params.page,
            "per_page": params.per_page
        }
        if params.type:
            query_params["type"] = params.type.value
        if params.name:
            query_params["name"] = params.name

        data = await _make_request("GET", f"zones/{zone_id}/dns_records", params=query_params)
        records = data.get("result", [])
        result_info = data.get("result_info", {})

        if not records:
            filter_msg = ""
            if params.type:
                filter_msg += f" type={params.type.value}"
            if params.name:
                filter_msg += f" name={params.name}"
            return f"No DNS records found for zone '{params.zone}'{filter_msg}."

        if params.response_format == ResponseFormat.JSON:
            return json.dumps({
                "zone": params.zone,
                "zone_id": zone_id,
                "records": records,
                "total": result_info.get("total_count", len(records)),
                "page": result_info.get("page", params.page)
            }, indent=2)

        # Markdown format
        lines = [f"# DNS Records for {params.zone}", ""]
        lines.append(f"Zone ID: `{zone_id}`")
        lines.append(f"Showing {len(records)} of {result_info.get('total_count', len(records))} records")
        lines.append("")

        # Group by record type
        by_type = {}
        for record in records:
            rtype = record["type"]
            if rtype not in by_type:
                by_type[rtype] = []
            by_type[rtype].append(record)

        for rtype in sorted(by_type.keys()):
            lines.append(f"## {rtype} Records")
            lines.append("")
            for record in by_type[rtype]:
                lines.append(_format_record_markdown(record))
                lines.append("")

        result = "\n".join(lines)
        if len(result) > CHARACTER_LIMIT:
            result = result[:CHARACTER_LIMIT] + "\n\n**[Output truncated. Use filters to narrow results.]**"

        return result

    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="cloudflare_create_dns_record",
    annotations={
        "title": "Create DNS Record",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def cloudflare_create_dns_record(params: CreateDNSRecordInput) -> str:
    """Create a new DNS record in a zone.

    Args:
        params: CreateDNSRecordInput with record details

    Returns:
        Created record details including the new record ID

    Examples:
        - A record: type=A, name='www', content='192.168.1.1'
        - MX record: type=MX, name='@', content='mail.example.com', priority=10
        - TXT record: type=TXT, name='@', content='v=spf1 mx -all'
    """
    try:
        zone_id = await _resolve_zone_id(params.zone)

        record_data = {
            "type": params.type.value,
            "name": params.name,
            "content": params.content,
            "ttl": params.ttl,
            "proxied": params.proxied
        }

        if params.priority is not None:
            record_data["priority"] = params.priority
        if params.comment:
            record_data["comment"] = params.comment

        data = await _make_request("POST", f"zones/{zone_id}/dns_records", json_data=record_data)
        record = data.get("result", {})

        return f"DNS record created successfully:\n\n{_format_record_markdown(record)}"

    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="cloudflare_update_dns_record",
    annotations={
        "title": "Update DNS Record",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def cloudflare_update_dns_record(params: UpdateDNSRecordInput) -> str:
    """Update an existing DNS record. Use cloudflare_list_dns_records to get record IDs.

    Args:
        params: UpdateDNSRecordInput with record ID and fields to update

    Returns:
        Updated record details
    """
    try:
        zone_id = await _resolve_zone_id(params.zone)

        # Build update payload with only provided fields
        update_data = {}
        if params.type is not None:
            update_data["type"] = params.type.value
        if params.name is not None:
            update_data["name"] = params.name
        if params.content is not None:
            update_data["content"] = params.content
        if params.ttl is not None:
            update_data["ttl"] = params.ttl
        if params.priority is not None:
            update_data["priority"] = params.priority
        if params.proxied is not None:
            update_data["proxied"] = params.proxied
        if params.comment is not None:
            update_data["comment"] = params.comment

        if not update_data:
            return "Error: No fields to update. Provide at least one field to change."

        data = await _make_request(
            "PATCH",
            f"zones/{zone_id}/dns_records/{params.record_id}",
            json_data=update_data
        )
        record = data.get("result", {})

        return f"DNS record updated successfully:\n\n{_format_record_markdown(record)}"

    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="cloudflare_delete_dns_record",
    annotations={
        "title": "Delete DNS Record",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def cloudflare_delete_dns_record(params: DeleteDNSRecordInput) -> str:
    """Delete a DNS record. This action cannot be undone.

    Use cloudflare_list_dns_records first to find the record ID.

    Args:
        params: DeleteDNSRecordInput with zone and record ID

    Returns:
        Confirmation of deletion
    """
    try:
        zone_id = await _resolve_zone_id(params.zone)

        data = await _make_request("DELETE", f"zones/{zone_id}/dns_records/{params.record_id}")

        if data.get("success"):
            return f"DNS record `{params.record_id}` deleted successfully from zone `{params.zone}`."
        else:
            return f"Failed to delete DNS record. Response: {json.dumps(data)}"

    except Exception as e:
        return _handle_error(e)


# =============================================================================
# Redirect Rules (Single Redirects)
# =============================================================================

_REDIRECT_PHASE = "http_request_dynamic_redirect"
_REDIRECT_ENTRYPOINT = "zones/{zid}/rulesets/phases/" + _REDIRECT_PHASE + "/entrypoint"


async def _get_redirect_entrypoint(zone_id: str):
    """Return (ruleset_id, rules[]) for the zone's dynamic-redirect entrypoint.

    Cloudflare returns 404 if no redirect rule has ever existed on the zone —
    we treat that as 'empty, none yet' rather than an error.
    """
    try:
        data = await _make_request("GET", _REDIRECT_ENTRYPOINT.format(zid=zone_id))
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None, []
        raise
    rs = data.get("result", {}) or {}
    return rs.get("id"), (rs.get("rules") or [])


async def _put_redirect_rules(zone_id: str, rules: list) -> list:
    """Replace the dynamic-redirect entrypoint's rules array (create-or-replace)."""
    data = await _make_request(
        "PUT", _REDIRECT_ENTRYPOINT.format(zid=zone_id), json_data={"rules": rules}
    )
    rs = data.get("result", {}) or {}
    return rs.get("rules") or []


def _build_from_value(target_url: str, status_code: int, preserve_query_string: bool, preserve_path: bool) -> dict:
    """Build action_parameters.from_value for a redirect rule."""
    fv = {"status_code": status_code, "preserve_query_string": preserve_query_string}
    if preserve_path:
        # Dynamic redirect: append the incoming path to the target.
        safe_target = target_url.replace('"', '\\"')
        fv["target_url"] = {"expression": f'concat("{safe_target}", http.request.uri.path)'}
    else:
        fv["target_url"] = {"value": target_url}
    return fv


def _format_redirect_rule_markdown(rule: dict) -> str:
    ap = (rule.get("action_parameters") or {})
    fv = (ap.get("from_value") or {})
    tu = fv.get("target_url") or {}
    target = tu.get("value") or tu.get("expression") or "?"
    lines = [
        f"### Redirect rule `{rule.get('id', '(new)')}`",
        f"- **Match**: `{rule.get('expression', '')}`",
        f"- **Target**: `{target}`",
        f"- **Status**: {fv.get('status_code', '?')}",
        f"- **Preserve query string**: {'Yes' if fv.get('preserve_query_string') else 'No'}",
        f"- **Enabled**: {'Yes' if rule.get('enabled', True) else 'No'}",
    ]
    if rule.get("description"):
        lines.append(f"- **Description**: {rule['description']}")
    return "\n".join(lines)


@mcp.tool(
    name="cloudflare_list_redirect_rules",
    annotations={"title": "List Redirect Rules", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
)
async def cloudflare_list_redirect_rules(params: ListRedirectRulesInput) -> str:
    """List a zone's Single Redirect rules (dynamic-redirect ruleset).

    Returns each rule's id, match expression, target, status code, and enabled
    state. Empty before any redirect has been created on the zone.
    """
    try:
        zone_id = await _resolve_zone_id(params.zone)
        ruleset_id, rules = await _get_redirect_entrypoint(zone_id)

        if params.response_format == ResponseFormat.JSON:
            return json.dumps({"zone": params.zone, "zone_id": zone_id, "ruleset_id": ruleset_id, "rules": rules}, indent=2)

        if not rules:
            return f"No redirect rules on zone '{params.zone}' yet."
        lines = [f"# Redirect Rules for {params.zone}", "", f"Ruleset ID: `{ruleset_id}` · {len(rules)} rule(s)", ""]
        for rule in rules:
            lines.append(_format_redirect_rule_markdown(rule))
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="cloudflare_create_redirect_rule",
    annotations={"title": "Create Redirect Rule", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True}
)
async def cloudflare_create_redirect_rule(params: CreateRedirectRuleInput) -> str:
    """Create a Single Redirect (e.g. cbrc.events -> https://cbrcevents.net, 301).

    Reads the zone's dynamic-redirect entrypoint, appends the new rule, and PUTs
    the full rules array back (creates the ruleset if it didn't exist).

    Requires the API token to have Zone -> Dynamic Redirect -> Edit.
    """
    try:
        match = params.match_expression
        if not match and params.source_hostname:
            match = f'(http.host eq "{params.source_hostname}")'
        if not match:
            return "Error: provide match_expression or source_hostname so the rule knows what to match."

        zone_id = await _resolve_zone_id(params.zone)
        _, rules = await _get_redirect_entrypoint(zone_id)

        new_rule = {
            "action": "redirect",
            "action_parameters": {
                "from_value": _build_from_value(
                    params.target_url, params.status_code, params.preserve_query_string, params.preserve_path
                )
            },
            "expression": match,
            "enabled": params.enabled,
        }
        if params.description:
            new_rule["description"] = params.description

        rules = list(rules) + [new_rule]
        updated = await _put_redirect_rules(zone_id, rules)
        created = updated[-1] if updated else new_rule
        return f"Redirect rule created on '{params.zone}':\n\n{_format_redirect_rule_markdown(created)}"
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="cloudflare_update_redirect_rule",
    annotations={"title": "Update Redirect Rule", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
)
async def cloudflare_update_redirect_rule(params: UpdateRedirectRuleInput) -> str:
    """Update a redirect rule by id (only the fields you pass change).

    Read-modify-write on the dynamic-redirect entrypoint.
    """
    try:
        zone_id = await _resolve_zone_id(params.zone)
        _, rules = await _get_redirect_entrypoint(zone_id)
        rules = list(rules)

        idx = next((i for i, r in enumerate(rules) if r.get("id") == params.rule_id), None)
        if idx is None:
            return f"Error: redirect rule `{params.rule_id}` not found on zone '{params.zone}'."

        rule = dict(rules[idx])
        # Match expression
        if params.match_expression is not None:
            rule["expression"] = params.match_expression
        elif params.source_hostname is not None:
            rule["expression"] = f'(http.host eq "{params.source_hostname}")'
        # Enabled / description
        if params.enabled is not None:
            rule["enabled"] = params.enabled
        if params.description is not None:
            rule["description"] = params.description
        # from_value (rebuild if any of its components changed)
        cur_fv = ((rule.get("action_parameters") or {}).get("from_value") or {})
        cur_tu = cur_fv.get("target_url") or {}
        cur_is_dynamic = "expression" in cur_tu
        if any(v is not None for v in (params.target_url, params.status_code, params.preserve_query_string, params.preserve_path)):
            # Derive current target if target_url not supplied
            if params.target_url is not None:
                target = params.target_url
            elif cur_is_dynamic:
                # strip concat("...", http.request.uri.path) back to the literal
                expr = cur_tu.get("expression", "")
                target = expr.split('"')[1] if '"' in expr else expr
            else:
                target = cur_tu.get("value", "")
            status = params.status_code if params.status_code is not None else cur_fv.get("status_code", 301)
            pqs = params.preserve_query_string if params.preserve_query_string is not None else bool(cur_fv.get("preserve_query_string", True))
            ppath = params.preserve_path if params.preserve_path is not None else cur_is_dynamic
            rule["action"] = "redirect"
            rule["action_parameters"] = {"from_value": _build_from_value(target, status, pqs, ppath)}

        rules[idx] = rule
        updated = await _put_redirect_rules(zone_id, rules)
        out = next((r for r in updated if r.get("id") == params.rule_id), rule)
        return f"Redirect rule updated on '{params.zone}':\n\n{_format_redirect_rule_markdown(out)}"
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="cloudflare_delete_redirect_rule",
    annotations={"title": "Delete Redirect Rule", "readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True}
)
async def cloudflare_delete_redirect_rule(params: DeleteRedirectRuleInput) -> str:
    """Delete a redirect rule by id (read-modify-write on the entrypoint)."""
    try:
        zone_id = await _resolve_zone_id(params.zone)
        _, rules = await _get_redirect_entrypoint(zone_id)
        rules = list(rules)
        remaining = [r for r in rules if r.get("id") != params.rule_id]
        if len(remaining) == len(rules):
            return f"Error: redirect rule `{params.rule_id}` not found on zone '{params.zone}'."
        await _put_redirect_rules(zone_id, remaining)
        return f"Redirect rule `{params.rule_id}` deleted from zone '{params.zone}'. {len(remaining)} rule(s) remain."
    except Exception as e:
        return _handle_error(e)


# =============================================================================
# Cloudflare Pages
# =============================================================================


async def _resolve_account_id(account: Optional[str] = None) -> str:
    """Resolve an account name or ID to an account ID.

    If `account` looks like an ID (32 hex), return it. Otherwise GET /accounts
    and match by name/id, or auto-pick when the token sees exactly one account.
    """
    if account and len(account) == 32 and all(c in "0123456789abcdef" for c in account.lower()):
        return account
    data = await _make_request("GET", "accounts")
    accounts = data.get("result", []) or []
    if account:
        for a in accounts:
            if a.get("id") == account or a.get("name", "").lower() == account.lower():
                return a["id"]
        raise ValueError(f"Cloudflare account '{account}' not found with this token.")
    if not accounts:
        raise ValueError("No Cloudflare accounts accessible — the token needs Account-level access (e.g. Cloudflare Pages: Edit).")
    if len(accounts) > 1:
        names = ", ".join(f"{a.get('name','?')} ({a.get('id')})" for a in accounts)
        raise ValueError(f"Multiple accounts accessible: {names}. Pass account=<name or id>.")
    return accounts[0]["id"]


def _format_pages_project_markdown(p: dict) -> str:
    src = p.get("source") or {}
    cfg = src.get("config") or {}
    bc = p.get("build_config") or {}
    latest = p.get("latest_deployment") or {}
    lines = [
        f"### {p.get('name')}",
        f"- **Subdomain**: `{p.get('subdomain', '')}`",
        f"- **Production branch**: {p.get('production_branch', '')}",
    ]
    if src.get("type"):
        lines.append(f"- **Source**: {src.get('type')} → `{cfg.get('owner', '?')}/{cfg.get('repo_name', '?')}`")
    if bc.get("build_command"):
        lines.append(f"- **Build**: `{bc.get('build_command')}` → `{bc.get('destination_dir', '')}`")
    domains = p.get("domains") or []
    if domains:
        lines.append(f"- **Domains**: {', '.join(domains)}")
    if latest:
        stage = (latest.get("latest_stage") or {})
        lines.append(f"- **Latest deploy**: {latest.get('url', '')} ({stage.get('name', '?')}/{stage.get('status', '?')})")
    return "\n".join(lines)


def _format_pages_deployment_markdown(d: dict) -> str:
    stage = (d.get("latest_stage") or {})
    dt = (d.get("deployment_trigger") or {}).get("metadata") or {}
    return "\n".join([
        f"### {d.get('id', '')[:8]} — {d.get('environment', '')}",
        f"- **URL**: {d.get('url', '')}",
        f"- **Status**: {stage.get('name', '?')}/{stage.get('status', '?')}",
        f"- **Branch**: {dt.get('branch', '?')} · commit `{(dt.get('commit_hash') or '')[:8]}`",
        f"- **Created**: {d.get('created_on', '')}",
    ])


@mcp.tool(
    name="cloudflare_list_pages_projects",
    annotations={"title": "List Pages Projects", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
)
async def cloudflare_list_pages_projects(params: ListPagesProjectsInput) -> str:
    """List Cloudflare Pages projects in the account."""
    try:
        account_id = await _resolve_account_id(params.account)
        data = await _make_request("GET", f"accounts/{account_id}/pages/projects")
        projects = data.get("result", []) or []
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({"account_id": account_id, "projects": projects}, indent=2)
        if not projects:
            return "No Pages projects in this account."
        lines = ["# Cloudflare Pages Projects", "", f"{len(projects)} project(s)", ""]
        for p in projects:
            lines.append(_format_pages_project_markdown(p))
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="cloudflare_get_pages_project",
    annotations={"title": "Get Pages Project", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
)
async def cloudflare_get_pages_project(params: GetPagesProjectInput) -> str:
    """Get one Cloudflare Pages project's details (source, build config, domains, latest deploy)."""
    try:
        account_id = await _resolve_account_id(params.account)
        data = await _make_request("GET", f"accounts/{account_id}/pages/projects/{params.project_name}")
        p = data.get("result", {}) or {}
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(p, indent=2)
        return _format_pages_project_markdown(p)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="cloudflare_create_pages_project",
    annotations={"title": "Create Pages Project", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True}
)
async def cloudflare_create_pages_project(params: CreatePagesProjectInput) -> str:
    """Create a Cloudflare Pages project — optionally git-connected to a GitHub repo.

    For a GitHub-connected project pass github_owner + github_repo (and optionally
    build_command/destination_dir/root_dir). NOTE: the 'Cloudflare Workers and
    Pages' GitHub App must already be authorized for this account in the
    dashboard (a one-time step the API can't perform); otherwise creation fails
    with a connection error. Without github_*, a direct-upload project is created.
    """
    try:
        account_id = await _resolve_account_id(params.account)
        body: dict = {"name": params.name, "production_branch": params.production_branch}
        if params.github_owner and params.github_repo:
            body["source"] = {
                "type": "github",
                "config": {
                    "owner": params.github_owner,
                    "repo_name": params.github_repo,
                    "production_branch": params.production_branch,
                    "production_deployments_enabled": True,
                    "deployments_enabled": True,
                },
            }
        bc = {}
        if params.build_command:
            bc["build_command"] = params.build_command
        if params.destination_dir:
            bc["destination_dir"] = params.destination_dir
        if params.root_dir:
            bc["root_dir"] = params.root_dir
        if bc:
            body["build_config"] = bc

        data = await _make_request("POST", f"accounts/{account_id}/pages/projects", json_data=body)
        p = data.get("result", {}) or {}
        return f"Pages project created:\n\n{_format_pages_project_markdown(p)}"
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="cloudflare_delete_pages_project",
    annotations={"title": "Delete Pages Project", "readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True}
)
async def cloudflare_delete_pages_project(params: DeletePagesProjectInput) -> str:
    """Delete a Cloudflare Pages project. This cannot be undone."""
    try:
        account_id = await _resolve_account_id(params.account)
        data = await _make_request("DELETE", f"accounts/{account_id}/pages/projects/{params.project_name}")
        if data.get("success"):
            return f"Pages project `{params.project_name}` deleted."
        return f"Failed to delete project. Response: {json.dumps(data)}"
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="cloudflare_list_pages_deployments",
    annotations={"title": "List Pages Deployments", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
)
async def cloudflare_list_pages_deployments(params: ListPagesDeploymentsInput) -> str:
    """List a Pages project's deployments (newest first), with status + branch/commit."""
    try:
        account_id = await _resolve_account_id(params.account)
        data = await _make_request("GET", f"accounts/{account_id}/pages/projects/{params.project_name}/deployments")
        deployments = data.get("result", []) or []
        if params.response_format == ResponseFormat.JSON:
            return json.dumps({"project": params.project_name, "deployments": deployments}, indent=2)
        if not deployments:
            return f"No deployments for Pages project '{params.project_name}'."
        lines = [f"# Deployments — {params.project_name}", "", f"{len(deployments)} deployment(s)", ""]
        for d in deployments[:25]:
            lines.append(_format_pages_deployment_markdown(d))
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="cloudflare_create_pages_deployment",
    annotations={"title": "Create Pages Deployment", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True}
)
async def cloudflare_create_pages_deployment(params: CreatePagesDeploymentInput) -> str:
    """Trigger a new deployment for a git-connected Pages project (defaults to production).

    The project's repo + account must already be authorized on the Cloudflare
    Pages dashboard (git connection). Useful to re-deploy/rollback after a push.
    """
    try:
        account_id = await _resolve_account_id(params.account)
        body = {"branch": params.branch} if params.branch else None
        data = await _make_request(
            "POST", f"accounts/{account_id}/pages/projects/{params.project_name}/deployments", json_data=body
        )
        d = data.get("result", {}) or {}
        return f"Deployment triggered for '{params.project_name}':\n\n{_format_pages_deployment_markdown(d)}"
    except Exception as e:
        return _handle_error(e)


def main():
    """Entry point for cloudflare-mcp command."""
    mcp.run()


if __name__ == "__main__":
    main()
