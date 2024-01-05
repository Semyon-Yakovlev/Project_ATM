def into_json(
    org_id,
    name,
    org_type,
    address,
    street,
    coords,
    rating,
    reviews_count,
    reviews_rating,
):
    """Шаблон файла OUTPUT.json"""

    data_grabbed = {
        "ID": org_id,
        "org_type": org_type,
        "name": name,
        "address": address,
        "street": street,
        "coords": coords,
        "rating": rating,
        "reviews_count": reviews_count,
        "reviews_rating": reviews_rating,
    }
    return data_grabbed
