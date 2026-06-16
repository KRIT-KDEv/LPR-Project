from result_utils import (
    create_analysis_dir,
    save_result_json,
    load_result_json,
    load_history,
    make_base_result,
    build_province_counts,
    build_timeline_from_tracks,
)


def main():
    analysis_dir = create_analysis_dir("test")

    result = make_base_result(
        analysis_id=analysis_dir.name,
        file_type="video",
        file_name="test_car_video.mp4",
        output_path=str(analysis_dir / "output_tracked_video.mp4"),
    )

    result["tracks"] = [
        {
            "track_id": 1,
            "plate_text": "กท2456,สุรินทร์",
            "plate_number": "กท2456",
            "province": "สุรินทร์",
            "vote_count": 2,
            "best_confidence": 0.92,
            "first_seen_sec": 3.2,
            "last_seen_sec": 5.4,
        },
        {
            "track_id": 2,
            "plate_text": "ฎ2768,กรุงเทพมหานคร",
            "plate_number": "ฎ2768",
            "province": "กรุงเทพมหานคร",
            "vote_count": 1,
            "best_confidence": 0.90,
            "first_seen_sec": 7.0,
            "last_seen_sec": 9.1,
        },
    ]

    result["province_counts"] = build_province_counts(result["tracks"])
    result["timeline"] = build_timeline_from_tracks(result["tracks"])

    result_path = save_result_json(result, analysis_dir)

    print("Result saved:", result_path)

    loaded = load_result_json(result_path)

    print("\nLoaded result:")
    print(loaded)

    history = load_history()

    print("\nHistory count:", len(history))


if __name__ == "__main__":
    main()