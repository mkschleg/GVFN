### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 174b25fe-28fa-11eb-2a22-734aacc6b674
using Revise

# ╔═╡ acbb4c5c-28fc-11eb-02ac-39a012b86a3c
using Plots, RollingFunctions, Pluto, Statistics, GVFN, LinearAlgebra

# ╔═╡ 4159b31c-28fd-11eb-2be8-33f6ee161358
function normalize_cols(mat::AbstractMatrix)
	ret = copy(mat)
	for i ∈ 1:size(mat)[2]
		s_max = maximum(ret[:,i])
		s_min = minimum(ret[:, i])
		# s = s == 0.0 ? 1.0 : s
		if s_max-s_min != 0.0
			ret[:, i] .= (ret[:, i] .- s_min)./(s_max-s_min)
		end
	end
	ret
end

# ╔═╡ 6fd21de4-2900-11eb-07e4-57220843db94
sensor_names = GVFN.CritterbotUtils.getSensorNames()

# ╔═╡ d035a6e2-2900-11eb-2340-d7cadfb52a24
relevant_sensors = GVFN.CritterbotUtils.relevant_sensor_idx()

# ╔═╡ 8bee15ec-28fa-11eb-0794-abd5c4d62ded
sensors = GVFN.CritterbotUtils.loadSensor_cmajor(relevant_sensors)

# ╔═╡ a14602bc-28fd-11eb-3581-1d08e9c9a2bd
norm_sensors = normalize_cols(sensors)

# ╔═╡ f1060ff6-28fe-11eb-09bc-4fe17756de15
work_data = norm_sensors;

# ╔═╡ bf53877e-28fe-11eb-37c7-d16092cd7bd5
mean_sensors = mean(work_data; dims=1)

# ╔═╡ 06353b56-28ff-11eb-09d2-65fa579d29c9
heatmap(abs.(cor(work_data)))

# ╔═╡ 024c4882-2901-11eb-3b2b-f109fc3de5d2
rets = normalize_cols(GVFN.CritterbotUtils.getReturns(relevant_sensors, 0.9975)')

# ╔═╡ 9a273a06-2906-11eb-3377-bddf485a3e9e
heatmap(sensor_names[relevant_sensors], sensor_names[relevant_sensors], abs.(cor(rets)), xrotation = 90, size=(1200,800), xticks=:all, yticks=:all)

# ╔═╡ 7588385e-2914-11eb-04b8-356087fe63c8
savefig("returns_heatmap.pdf")

# ╔═╡ cfe613a6-2901-11eb-1207-1fad0bf09a2e
@bind range_max html"<input type='range'>"

# ╔═╡ 4ce45ee6-290f-11eb-095e-9784c9be38f6
@bind sensor html"<input type='number' default='1' id='quantity' name='quantity' min='1' max='93'>"

# ╔═╡ 4df8e094-2901-11eb-1fd4-4747788cd5b0
@show range_max

# ╔═╡ 577d516c-2902-11eb-280f-2f2e543329e2
@show sensor

# ╔═╡ 98b85680-290f-11eb-38bd-d9efa430538b
begin
	rm = Int(range_max/100 * (size(rets)[1] - 20000) + 1)
	range = rm:(rm+19999)
	plot(norm_sensors[range, sensor], label="Sensor", alpha=0.4)
	plot!(rets[range, sensor], label="Return")
end

# ╔═╡ 531c631e-2915-11eb-27ea-9313a76de394
target_sens, obs_sens = ["Mag0", "Mag2"], ["Light1", "Light2", "Light3", "Thermal0"]

# ╔═╡ 732adf66-2915-11eb-0009-41ffc35fa94a
final_sensors = normalize_cols(GVFN.CritterbotUtils.loadSensor_cmajor(obs_sens))

# ╔═╡ b47ab02c-2915-11eb-2ed7-31fae87079d1
final_targets = normalize_cols(GVFN.CritterbotUtils.loadSensor_cmajor(target_sens))

# ╔═╡ ba9f87b0-2916-11eb-3878-970d78f737cb
@bind range_max_2 html"<input type='range'>"

# ╔═╡ c930495a-2915-11eb-0687-7f55d222cd5d
begin
	rm_2 = Int(range_max_2/100 * (size(final_sensors)[1] - 20000) + 1)
	range_2 = rm_2:(rm_2+19999)
	plot(final_sensors[range_2, 4], label="Sensor", alpha=0.4)
	# plot!(rets[range, sensor], label="Return")
end

# ╔═╡ 668953a8-291c-11eb-0eff-a3b4a8d3e141
function temporal_cor(x, y, horizon)
	R = cor(x)
	c = [cor(x[1:end-horizon,i], y[(horizon+1):end]) for i ∈ 1:(size(x)[2])]
	c'*R*c
end

# ╔═╡ b1f1eb44-291b-11eb-3aaf-5360c13eb28c
plot([temporal_cor(final_sensors, final_targets[:, 1], i) for i ∈ 1:64])

# ╔═╡ daa8ddd0-291c-11eb-2cc7-7125cfb2a094
plot([temporal_cor(final_sensors, final_targets[:, 2], i) for i ∈ 1:64])

# ╔═╡ ee1af876-291c-11eb-3f9e-7f06000548af
final_rets = normalize_cols(GVFN.CritterbotUtils.getReturns(obs_sens, [0.0, 0.5, 0.75, 0.9, 0.95, 0.99, 0.9975])')

# ╔═╡ 0b024c32-291d-11eb-3c2a-5d8af2bb51e8
plot([temporal_cor(final_rets, final_targets[:, 1], i) for i ∈ 1:64])

# ╔═╡ 563eb1e6-2921-11eb-31b6-b15bb20552ec
@bind sensor_cor html"<input type='number' default='1' id='quantity' name='quantity' min='1' max='93'>"

# ╔═╡ a472b114-2921-11eb-21c2-cf0261b232c7
plots = [plot([temporal_cor(final_rets, normalize(GVFN.CritterbotUtils.loadSensor(sc)), i) for i ∈ 1:64], label=sensor_names[sc+1]) for sc ∈ 1:93];

# ╔═╡ 6f3b586e-291e-11eb-1af2-8d1444ed18bb
plots[sensor_cor]

# ╔═╡ 8636c712-29b9-11eb-08d5-999617b110fc
plots_2 = begin
	newrets = ["Mag0", "Mag1", "Thermal0"]
	final_rets_2 = normalize_cols(GVFN.CritterbotUtils.getReturns(newrets, [0.0, 0.5, 0.75, 0.9, 0.95, 0.99, 0.9975])')
	[plot([temporal_cor(final_rets_2, normalize(GVFN.CritterbotUtils.loadSensor(sc)), i) for i ∈ 1:64], label=sensor_names[sc+1]) for sc ∈ 1:93];
end

# ╔═╡ bac65144-29c3-11eb-20e8-3d5297face93
@bind sensor_cor_2 html"<input type='number' default='1' id='quantity' name='quantity' min='1' max='93'>"

# ╔═╡ 21f3a6aa-29ba-11eb-25e3-e77acaa017f1
plots_2[sensor_cor_2]

# ╔═╡ Cell order:
# ╠═174b25fe-28fa-11eb-2a22-734aacc6b674
# ╠═acbb4c5c-28fc-11eb-02ac-39a012b86a3c
# ╠═4159b31c-28fd-11eb-2be8-33f6ee161358
# ╠═6fd21de4-2900-11eb-07e4-57220843db94
# ╠═d035a6e2-2900-11eb-2340-d7cadfb52a24
# ╠═8bee15ec-28fa-11eb-0794-abd5c4d62ded
# ╠═a14602bc-28fd-11eb-3581-1d08e9c9a2bd
# ╠═f1060ff6-28fe-11eb-09bc-4fe17756de15
# ╠═bf53877e-28fe-11eb-37c7-d16092cd7bd5
# ╠═06353b56-28ff-11eb-09d2-65fa579d29c9
# ╠═024c4882-2901-11eb-3b2b-f109fc3de5d2
# ╠═9a273a06-2906-11eb-3377-bddf485a3e9e
# ╠═7588385e-2914-11eb-04b8-356087fe63c8
# ╠═cfe613a6-2901-11eb-1207-1fad0bf09a2e
# ╠═4ce45ee6-290f-11eb-095e-9784c9be38f6
# ╠═4df8e094-2901-11eb-1fd4-4747788cd5b0
# ╠═577d516c-2902-11eb-280f-2f2e543329e2
# ╠═98b85680-290f-11eb-38bd-d9efa430538b
# ╠═531c631e-2915-11eb-27ea-9313a76de394
# ╠═732adf66-2915-11eb-0009-41ffc35fa94a
# ╠═b47ab02c-2915-11eb-2ed7-31fae87079d1
# ╠═ba9f87b0-2916-11eb-3878-970d78f737cb
# ╠═c930495a-2915-11eb-0687-7f55d222cd5d
# ╠═668953a8-291c-11eb-0eff-a3b4a8d3e141
# ╠═b1f1eb44-291b-11eb-3aaf-5360c13eb28c
# ╠═daa8ddd0-291c-11eb-2cc7-7125cfb2a094
# ╠═ee1af876-291c-11eb-3f9e-7f06000548af
# ╠═0b024c32-291d-11eb-3c2a-5d8af2bb51e8
# ╠═563eb1e6-2921-11eb-31b6-b15bb20552ec
# ╠═a472b114-2921-11eb-21c2-cf0261b232c7
# ╠═6f3b586e-291e-11eb-1af2-8d1444ed18bb
# ╠═8636c712-29b9-11eb-08d5-999617b110fc
# ╠═bac65144-29c3-11eb-20e8-3d5297face93
# ╠═21f3a6aa-29ba-11eb-25e3-e77acaa017f1
