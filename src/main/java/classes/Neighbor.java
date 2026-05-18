package classes;

import java.util.ArrayList;

/**
 * @class Neighbor
 * @brief Represents a single data entry (a neighbor) in the KNN algorithm.
 * This class stores the numerical features (values) and the associated
 * classification category (label) for a specific point in the dataset.
 */
public class Neighbor {

    /** @brief Internal storage for the numerical features of the data point. */
    private final ArrayList<Double> values;
    /** @brief The classification category or class name assigned to this point. */
    private final String label;

    /**
     * @brief Constructor for the Neighbor class.
     * @param values An ArrayList of integers representing the point's coordinates or features.
     * @param label A String representing the class or category of this data point.
     */
    public Neighbor(ArrayList<Double> values,  String label) {
        this.values  = values;
        this.label = label;
    }

    /**
     * @brief Retrieves the numerical features of the neighbor.
     * @return An ArrayList containing the integer values of the point.
     */
    public ArrayList<Double> getValues() { return values; }

    /**
     * @brief Retrieves the classification label of the neighbor.
     * @return A String containing the label/category.
     */
    public String getLabel() { return label; }
}
