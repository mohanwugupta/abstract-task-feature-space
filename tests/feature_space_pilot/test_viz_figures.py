from feature_space_pilot.viz.load import load_run


def test_plot_specs_keep_ids_full_labels_and_nonstacked_composition(viz_run_path):
    from feature_space_pilot.viz.figures import plotly_matrix, plotly_composition
    from feature_space_pilot.viz.derive import matrix_data, composition_rows

    run = load_run(viz_run_path)
    data = matrix_data(run)
    fig = plotly_matrix(run, data)
    assert len(fig.data[0].z) == 15
    assert fig.data[0].customdata[0][0][0] == data["item_ids"][0]
    assert fig.data[0].customdata[0][0][1] == data["feature_ids"][0]
    assert (
        fig.data[0].customdata[0][0][2] == run.features[data["feature_ids"][0]]["text"]
    )
    comp = plotly_composition(composition_rows(run))
    assert comp.layout.barmode == "group" and len(comp.data) == 3
    assert all(len(trace.y) == 6 for trace in comp.data)


def test_static_propositions_and_review_notes_stay_in_figure_body(viz_run_path):
    import matplotlib.pyplot as plt
    from feature_space_pilot.viz.figures import static_atlas, static_composition

    run = load_run(viz_run_path)
    fig, _, _ = static_atlas(run)
    assert all(ax.get_xlim() == (0, 1) and ax.get_ylim() == (0, 1) for ax in fig.axes)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for text in ax.texts:
            bounds = text.get_window_extent(renderer).transformed(
                fig.transFigure.inverted()
            )
            assert 0.1 < bounds.y0 < bounds.y1 < 0.85
    plt.close(fig)
    fig, _, _ = static_composition(run)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for text in ax.texts:
            if text.get_text().startswith("Calls "):
                assert (
                    text.get_window_extent(renderer)
                    .transformed(fig.transFigure.inverted())
                    .y0
                    > 0.12
                )
    plt.close(fig)
