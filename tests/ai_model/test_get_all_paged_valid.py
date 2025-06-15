"""Test to check the default and non default returns of the paged models endpoint."""
# ruff: noqa: S101
# ruff noqa: PLR2004

import math

import pytest
from fastapi import status
from fastapi.testclient import TestClient

SEEDED_MODEL_COUNT = 10  # Set to the actual number of seeded models


@pytest.mark.ai_model
def test_list_models_pagination_size_10(client: TestClient) -> None:
    """Test for non default page size."""
    page_size = 9
    page_number = 2
    last_pagenum = math.ceil(SEEDED_MODEL_COUNT / page_size)
    expected_last_page_items = SEEDED_MODEL_COUNT % page_size or page_size

    route = f"/models?page={page_number}&size={page_size}"
    resp = client.get(route)
    assert resp.status_code == status.HTTP_200_OK
    payload = resp.json()
    assert payload["page"] == page_number
    assert payload["size"] == page_size
    assert payload["totalpages"] == last_pagenum
    assert len(payload["items"]) == expected_last_page_items

    # Last page
    route = f"/models?page={last_pagenum}&size={page_size}"
    resp = client.get(route)
    assert resp.status_code == status.HTTP_200_OK
    payload = resp.json()
    assert payload["page"] == last_pagenum
    assert len(payload["items"]) == expected_last_page_items


@pytest.mark.ai_model
def test_list_models_pagination_size_5(client: TestClient) -> None:
    """Test for default page size."""
    page_size = 5
    page_number = 1
    last_pagenum = math.ceil(SEEDED_MODEL_COUNT / page_size)
    expected_last_page_items = SEEDED_MODEL_COUNT % page_size or page_size

    # First page
    route = f"/models?page={page_number}&size={page_size}"
    resp = client.get(route)
    assert resp.status_code == status.HTTP_200_OK
    payload = resp.json()
    assert payload["page"] == page_number
    assert payload["size"] == page_size
    assert payload["totalpages"] == last_pagenum
    assert len(payload["items"]) == page_size

    # Last page
    route = f"/models?page={last_pagenum}&size={page_size}"
    resp = client.get(route)
    assert resp.status_code == status.HTTP_200_OK
    payload = resp.json()
    assert payload["page"] == last_pagenum
    assert len(payload["items"]) == expected_last_page_items
