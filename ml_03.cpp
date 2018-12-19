#include <iostream>
#include <fstream>
#include <string>
#include <sstream> 
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>

struct similarity_record {
	size_t from, to;
	float value;
};

similarity_record extract_ids(std::string line, bool has_similarity=false)
{
	similarity_record record{0, 0, -1.};

	std::stringstream ss;
	ss << line;

	ss >> line;
	std::stringstream(line) >> record.from;
	ss >> line;
	std::stringstream(line) >> record.to;
	
	if (has_similarity) {
		ss >> line;
		std::stringstream(line) >> record.value;
	}

	return record;
}

void cache_exemplars(const char* file_name, const std::vector<size_t>& data) {
	std::ofstream write_stream;
	write_stream.open(file_name);

	if (write_stream.is_open()) {
		for (size_t i = 0; i < data.size(); ++i) {
			write_stream << i << "\t" << data.at(i) << std::endl;
		}
	}
}

void cache_top_results(const char* file_name, const std::vector<std::vector<int>>& data) {
	std::ofstream write_stream;
	write_stream.open(file_name);

	if (write_stream.is_open()) {
		for (size_t i = 0; i < data.size(); ++i) {
			write_stream << i << "\t";

			for (auto& k : data.at(i)) {
				write_stream << k << "\t";
			}

			write_stream << std::endl;
		}
	}
}

struct CEdgeAP {
	int from, to;
	float s, a, r;

	CEdgeAP(int from, int to): 
		from(from), to(to){
		this->a = this->r = this->s = 0;
	}
};

class CGraphAP {
	friend class CAffinityPropagation;

	using EdgeVector = std::vector<CEdgeAP*>;

	EdgeVector** pp_inputEdges = nullptr;
	EdgeVector** pp_outputEdges = nullptr;
	std::vector<size_t> exemplars;
	size_t i_matrixSize;

	std::default_random_engine generator;
	std::normal_distribution<float> norm_distr;
private:
	CEdgeAP* make_edge(int from, int to, float s = -1) {
		CEdgeAP* edge = new CEdgeAP(from, to);
		edge->s = s + this->next_random() / 1e-6;
		return edge;
	}

	CEdgeAP* make_edge_pref(int node_id) {
		CEdgeAP* edge = new CEdgeAP(node_id, node_id);
		edge->s = this->next_random() / 1e-6;
		return edge;
	}

	inline float next_random() {
		return this->norm_distr(this->generator) / 1e-3f;
	}

	inline void append_record(CEdgeAP* edge) {
		if (!this->pp_inputEdges[edge->to])
			this->pp_inputEdges[edge->to] = new EdgeVector();

		if (!this->pp_outputEdges[edge->from])
			this->pp_outputEdges[edge->from] = new EdgeVector();

		this->pp_inputEdges[edge->to]->push_back(edge);
		this->pp_outputEdges[edge->from]->push_back(edge);
	}

public:
	CGraphAP() {
		this->norm_distr = std::normal_distribution<float>(0., 1.0);
	}

	void read_from_file(const char* fname, size_t n_unique, bool has_similarity=false) {
		this->i_matrixSize = n_unique;

		std::ifstream file(fname);
		this->exemplars = std::vector<size_t>(n_unique, -1);

		this->pp_inputEdges = new EdgeVector*[n_unique] {nullptr};
		this->pp_outputEdges = new EdgeVector*[n_unique] {nullptr};

		if (file.is_open()) {
			std::cout << "loading data..." << std::endl;
			std::string line;
			size_t step_counter = 0;

			while (std::getline(file, line)) {
				if (!(++step_counter % 10000)) {
					std::cout << "processed " << step_counter << " records\r";
					std::cout.flush();
				}

				similarity_record extracted = extract_ids(line, has_similarity);
				this->append_record(this->make_edge(extracted.from, extracted.to, extracted.value));
			}

			// preferences
			for (size_t i = 0; i < this->i_matrixSize; ++i) {
				this->append_record(this->make_edge_pref(i));
			}

			file.close();
		}
	}
};

class CAffinityPropagation {
	CGraphAP* p_graph;
	float i_damping;
	size_t i_maxIter;

	inline void update_with_damping(float &_old, float _new) {
		_old = this->i_damping * _old + (1.f - this->i_damping) * (_new);
	}

	size_t get_unique_exemplars() {
		auto& exemplars = this->p_graph->exemplars;
		std::vector<size_t> exemplars_copy(exemplars.size());
		std::copy(exemplars.begin(), exemplars.end(), exemplars_copy.begin());
		std::sort(exemplars_copy.begin(), exemplars_copy.end());
		auto last = std::unique(exemplars_copy.begin(), exemplars_copy.end());
		exemplars_copy.erase(last, exemplars_copy.end());
		
		return exemplars_copy.size();
	}

	void update_responsibility() {
		for (size_t i = 0; i < p_graph->i_matrixSize; ++i) {
			auto& p_edgesRow = this->p_graph->pp_outputEdges[i];

			float first_maximum_value = -std::numeric_limits<float>::infinity();
			float second_maximum_value = -std::numeric_limits<float>::infinity();
			size_t maximum_index = -1;
			float temp_value;

			// find fist and second maximums
			for (size_t j = 0; j < p_edgesRow->size(); ++j) {
				auto& edge = p_edgesRow->at(j);
				temp_value = edge->s + edge->a;

				if (temp_value > first_maximum_value) {
					std::swap(first_maximum_value, temp_value);
					maximum_index = j;
				}

				if (temp_value > second_maximum_value)
					second_maximum_value = temp_value;
			}

			// update responsibilities
			for (size_t j = 0; j < maximum_index; ++j){
				auto& edge = p_edgesRow->at(j);
				this->update_with_damping(edge->r, edge->s - first_maximum_value);
			}

			this->update_with_damping(
				p_edgesRow->at(maximum_index)->r,
				p_edgesRow->at(maximum_index)->s - second_maximum_value
			);

			for (size_t j = maximum_index + 1; j < p_edgesRow->size(); ++j) {
				auto& edge = p_edgesRow->at(j);
				this->update_with_damping(edge->r, edge->s - first_maximum_value);
			}
		}
	}

	void update_availability() {
		for (size_t i = 0; i < p_graph->i_matrixSize; ++i) {
			auto& p_edgesRow = this->p_graph->pp_inputEdges[i];
			float cumsum = 0;

			for (size_t j = 0; j < p_edgesRow->size()-1; ++j)
				cumsum += std::max<float>(0., p_edgesRow->at(j)->r);

			// self reference is at the end of vector
			float self_responsibility = p_edgesRow->back()->r;

			// update non diagonal elements
			for (size_t j = 0; j < p_edgesRow->size() - 1; ++j)
				this->update_with_damping(
					p_edgesRow->at(j)->a,
					self_responsibility + cumsum - std::max<float>(0., p_edgesRow->at(j)->r)
				);
			
			// update diagonal elements
			this->update_with_damping(
				p_edgesRow->back()->a,
				cumsum
			);
		}
	}

	size_t update_exemplars() {
		size_t changes_count = 0;

		for (size_t i = 0; i < p_graph->i_matrixSize; ++i) {
			auto& p_edgesRow = this->p_graph->pp_outputEdges[i];

			float max_value = -std::numeric_limits<float>::infinity();
			size_t max_index = -1;

			for (size_t j = 0; j < p_edgesRow->size(); ++j) {
				auto& edge = p_edgesRow->at(j);
				float tmp_val = edge->a + edge->r;

				if(tmp_val > max_value){
					max_value = tmp_val;
					max_index = p_edgesRow->at(j)->to;
				}
			}

			auto& exemplars = this->p_graph->exemplars;

			if (exemplars[i] != max_index) {
				changes_count++;
				exemplars[i] = max_index;
			}
		}

		return changes_count;
	}

public:
	CAffinityPropagation(float damping, size_t i_maxIter) : i_damping(damping), i_maxIter(i_maxIter) {}

	std::vector<size_t>& fit_predict(CGraphAP& graph) {
		std::cout << "fitting affinity propagation" << std::endl;
		this->p_graph = &graph;

		size_t cummulated_changes = 0;
		for (size_t i = 1; i <= this->i_maxIter; ++i) {

			this->update_responsibility();
			this->update_availability();

			size_t changes_count = this->update_exemplars();
			size_t unique_exemplars = this->get_unique_exemplars();
			cummulated_changes += changes_count;

			if (i % 10 == 0) {
				std::cout << "AP iteration (" << i << "/" << this->i_maxIter << "), exemplars :" << unique_exemplars << ", changes:" << cummulated_changes << std::endl;
				cummulated_changes = 0;
			}

			if (changes_count == 0) {
				std::cout << "execution interrupted at " << i << " due to convergence" << std::endl;
				std::cout << "total number of exemplars: " << this->get_unique_exemplars() << std::endl;
				break;
			}
		}

		return this->p_graph->exemplars;
	}

	std::vector<std::vector<int>> get_top_candidates(int n_candidates) {
		std::vector<std::vector<int>> result;

		for (size_t i = 0; i < p_graph->i_matrixSize; ++i) {
			auto& p_edgesRow = this->p_graph->pp_outputEdges[i];

			float* node_top = new float[n_candidates] {-std::numeric_limits<float>::infinity()};
			int* node_ix_top = new int[n_candidates]{ -1 };

			for (size_t j = 0; j < p_edgesRow->size(); ++j) {
				auto& edge = p_edgesRow->at(j);
				float tmp_val = edge->a + edge->r;

				int current_index = j;
				for (size_t k = 0; k < n_candidates; ++k) {
					if (node_top[k] < tmp_val) {
						std::swap(node_top[k], tmp_val);
						std::swap(node_ix_top[k], current_index);
					}
				}
			}

			result.push_back(std::vector<int>(node_ix_top, node_ix_top + n_candidates));

			delete[] node_top;
			delete[] node_ix_top;
		}

		return result;
	}
};

void main() {
	const size_t total_nodes = 196591;
	const size_t iterations = 500;

	auto& graph = CGraphAP();
	auto& affinity_propagation = CAffinityPropagation(.6, iterations);

	//graph.read_from_file("C:/Users/Petr/JUPYTER/Gowalla/Gowalla_edges.txt", total_nodes);
	graph.read_from_file("C:/Users/Petr/JUPYTER/Gowalla/synth.txt", 100, true);
	
	auto& exemplars = affinity_propagation.fit_predict(graph);
	//auto& predictions = affinity_propagation.get_top_candidates(10);
	

	cache_exemplars("C:/Users/Petr/JUPYTER/Gowalla/exemplars.txt", exemplars);
	//cache_top_results("C:/Users/Petr/JUPYTER/Gowalla/exemplars.txt", predictions);
	std::getchar();
}